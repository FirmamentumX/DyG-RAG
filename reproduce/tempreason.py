from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, time, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from openai import AsyncOpenAI
import aiohttp.client_exceptions
import torch, gc

from graphrag import GraphRAG, QueryParam
from graphrag.base import BaseKVStorage
from graphrag._utils import compute_args_hash, logger



WORK_DIR = Path("tempreason_dir")
WORK_DIR.mkdir(exist_ok=True)
CORPUS_FILE = Path("datasets/TempReason/Corpus.json")

QUERY_MODE = "dynamic" 
QUERY_TOP_K = 20
CONCURRENCY = 5

QUESTIONS_FILE = Path("datasets/TempReason/Question.json")
RESULTS_FILE = Path(f"results_mode-{QUERY_MODE}_topk-{QUERY_TOP_K}.json")



import tiktoken
tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logging.getLogger("DyG-RAG").setLevel(logging.INFO)

################################################################################
# 0. Configuration
################################################################################
def get_config_value(env_name: str, description: str, example: str = None) -> str:
    """Get configuration value from environment variable or user input."""
    value = os.getenv(env_name)
    if value:
        return value
    
    print(f"\n⚠️  Missing configuration: {env_name}")
    print(f"Description: {description}")
    if example:
        print(f"Example: {example}")
    
    while True:
        user_input = input(f"Please enter {env_name}: ").strip()
        if user_input:
            return user_input
        print("❌ Value cannot be empty. Please try again.")

print("🔧 Checking configuration...")
VLLM_BASE_URL = get_config_value(
    "VLLM_BASE_URL", 
    "Base URL for VLLM API service", 
    "http://127.0.0.1:8000/v1"
)

BEST_MODEL_NAME = get_config_value(
    "QWEN_BEST", 
    "Model name for the best/primary LLM", 
    "qwen-14b"
)

LOCAL_BGE_PATH = get_config_value(
    "LOCAL_BGE_PATH", 
    "Local path to BGE embedding model", 
    "/path/to/bge-m3"
)

OPENAI_API_KEY_FAKE = "EMPTY"

################################################################################
# 1. Embedding function
################################################################################
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model: SentenceTransformer

    async def __call__(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        encode = lambda: self.model.encode(
            texts,
            batch_size=32,  # Smaller batch size to help with memory issues
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        if loop.is_running():
            return await loop.run_in_executor(None, encode)
        else:
            return encode()
    
    # Make the model not serializable for GraphRAG initialization
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state
    
    # Restore model reference during deserialization (will be None, but structure preserved)
    def __setstate__(self, state):
        self.__dict__.update(state)

def get_bge_embedding_func() -> EmbeddingFunc:
    gpu_count = torch.cuda.device_count()
    using_cuda = gpu_count > 0
    device = "cuda" if using_cuda else "cpu"

    model_kwargs = {}
    if gpu_count > 1:
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    st_model = SentenceTransformer(
        LOCAL_BGE_PATH,
        device=device,              
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )

    return EmbeddingFunc(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        max_token_size=8192,       
        model=st_model,
    )
    
################################################################################
# 2. LLM call function (with cache)
################################################################################
def _build_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=OPENAI_API_KEY_FAKE, base_url=VLLM_BASE_URL)

async def _chat_completion(model: str, messages: list[dict[str, str]], **kwargs) -> str:
    client = _build_async_client()
    response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content

async def _llm_with_cache(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    hashing_kv: BaseKVStorage | None = None,
    **kwargs,
) -> str:
    """General LLM wrapper, supporting GraphRAG cache interface."""
    history_messages = history_messages or []
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history_messages)
    msgs.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, msgs)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    answer = await _chat_completion(model=model, messages=msgs, **kwargs)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": answer, "model": model}})
        await hashing_kv.index_done_callback()
    return answer

async def best_model_func(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str,str]] | None = None, **kwargs) -> str:
    return await _llm_with_cache(
        prompt,
        model=BEST_MODEL_NAME,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

################################################################################
# 2. DyG-RAG initialization
################################################################################
embedding_func = get_bge_embedding_func()
model_ref = embedding_func.model
embedding_func.model = None 

graph_rag = GraphRAG(
    working_dir=str(WORK_DIR),
    embedding_func=embedding_func,
    best_model_func=best_model_func,
    cheap_model_func=best_model_func,
    enable_llm_cache=True,
    best_model_max_token_size = 16384,
    cheap_model_max_token_size = 16384,
    model_path="./models",  
    ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",  
    ner_model_name="dslim_bert_base_ner", 
)

embedding_func.model = model_ref

################################################################################
# 3. Insert corpus
################################################################################
assert CORPUS_FILE.exists(), f"{CORPUS_FILE} not found"
def load_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0) 
            for line in f:
                line = line.strip()
                if line: 
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Warning: Unable to parse line: {line[:50]}... Error: {e}")
            return data

start = time.time()
# Load all documents
logger.info(f"Starting to load documents from {CORPUS_FILE}...")
try:
    corpus_data = load_json_file(CORPUS_FILE)
    total_docs = len(corpus_data)
    logger.info(f"Successfully loaded {total_docs} documents")

    # Prepare all documents
    all_docs = []
    for idx, obj in enumerate(tqdm(corpus_data, desc="Processing documents", total=total_docs)):
        # Merge metadata and content
        enriched_content = f"Title: {obj['title']}\nDocument ID: {idx}\n\n{obj['context']}"
        all_docs.append(enriched_content)
    
    logger.info(f"Starting to insert all {len(all_docs)} documents into GraphRAG...")
    
    # Process all documents at once
    start_process_time = time.time()
    try:
        graph_rag.insert(all_docs)
        process_time = time.time() - start_process_time
        logger.info(f"All documents processed successfully! Time: {process_time:.1f} seconds, Average per document: {process_time/len(all_docs):.2f} seconds")
        processed_docs = len(all_docs)
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        import traceback
        traceback.print_exc()
        processed_docs = 0
    
    # Calculate total statistics
    elapsed = time.time() - start
    logger.info(f"Processing completed! Successfully processed: {processed_docs}/{total_docs} documents ({processed_docs/total_docs*100:.1f}%)")
    logger.info(f"Total time: {elapsed:.1f} seconds, Average per document: {elapsed/processed_docs:.2f} seconds (if any processing failed)")
    
except Exception as e:
    logger.error(f"Error loading documents: {str(e)}")
    import traceback
    traceback.print_exc()
    processed_docs = 0

# Ensure final persistence
logger.info("Executing final persistence...")
# GraphRAG does not have a persist method, use synchronous call method
loop = asyncio.get_event_loop()
loop.run_until_complete(graph_rag._insert_done())  

total_time = time.time() - start
logger.info(f"Processing completed! Total time {total_time:.1f} seconds")
if processed_docs > 0:
    logger.info(f"Successfully inserted {processed_docs}/{total_docs} documents, Average per document: {total_time/processed_docs:.2f} seconds")

################################################################################
# 4. Query test set and save results
################################################################################
import aiohttp.client_exceptions
timeout_exceptions = (
    asyncio.TimeoutError,
    aiohttp.client_exceptions.ClientConnectorError,
    aiohttp.client_exceptions.ServerTimeoutError,
    aiohttp.client_exceptions.ClientOSError,
    ConnectionRefusedError,
    ConnectionError,
    TimeoutError,
)

# Retry settings
max_retries = 5
retry_delay = 30  # seconds


async def process_single_query(question_obj, query_idx, graph_rag):
    """Process a single query and return the result"""
    try:
        # Check necessary fields
        if "question" not in question_obj:
            logger.error(f"Query {query_idx} missing question field: {str(question_obj)[:100]}...")
            return {
                "question_id": question_obj.get("id", f"q{query_idx}"),
                "error": "Missing question field",
                "status": "invalid_format"
            }
            
        question_id = question_obj.get("id", f"q{query_idx}")
        question_text = question_obj["question"]
        
        logger.info(f"Processing question {question_id}: {question_text[:50]}...")
        
        # Execute query with specified mode
        query_param = QueryParam(
            top_k=QUERY_TOP_K,
            mode=QUERY_MODE,
            response_type="Short Answer or Direct Response"
        )
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                result = await graph_rag.aquery(question_text, param=query_param)
                query_time = time.time() - start_time
                break
            except timeout_exceptions as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Query {question_id} timed out, retrying ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Query {question_id} failed after {max_retries} attempts: {str(e)}")
                    return {
                        "question_id": question_id,
                        "question": question_text,
                        "answer": None,
                        "error": str(e),
                        "query_time": time.time() - start_time,
                        "status": "failed"
                    }
        
        # Return results
        return {
            "question_id": question_id,
            "question": question_text,
            "answer": result,
            "query_time": query_time,
            "status": "success",
            "golden_answer": question_obj.get("answer", "")
        }
    except Exception as e:
        logger.error(f"Error processing question {query_idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "question_id": question_obj.get("id", f"q{query_idx}"),
            "question": question_obj.get("question", ""),
            "answer": None,
            "error": str(e),
            "status": "exception",
            "traceback": traceback.format_exc()
        }


async def process_timeqa_queries(graph_rag):
    """Process TempReason questions and save results"""
    logger.info(f"Loading TempReason questions from {QUESTIONS_FILE}...")
    
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        total_questions = len(questions)
        logger.info(f"Successfully loaded {total_questions} questions")
        
        valid_questions = sum(1 for q in questions if "question" in q and "answer" in q)
        if valid_questions < total_questions:
            logger.warning(f"{total_questions - valid_questions} questions missing required fields!")
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}")
        return []
    
    results = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def bounded_process(question, idx):
        async with semaphore:
            return await process_single_query(question, idx, graph_rag)
    
    # Create tasks
    tasks = [bounded_process(question, i) for i, question in enumerate(questions)]
    
    start_time = time.time()
    for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing TempReason queries"):
        result = await completed_task
        results.append(result)
        
        if len(results) % 5 == 0:
            with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "total_questions": total_questions,
                        "completed_questions": len(results),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "query_mode": QUERY_MODE,
                        "query_top_k": QUERY_TOP_K
                    },
                    "results": results
                }, f, ensure_ascii=False, indent=2)
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_questions": total_questions,
                "completed_questions": len(results),
                "total_time": time.time() - start_time,
                "avg_time_per_question": (time.time() - start_time) / len(results) if results else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query_mode": QUERY_MODE,
                "query_top_k": QUERY_TOP_K
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    exception_count = sum(1 for r in results if r["status"] == "exception")
    
    logger.info(f"TempReason query processing completed!")
    logger.info(f"Total: {total_questions}, Success: {success_count}, Failed: {failed_count}, Exception: {exception_count}")
    logger.info(f"Results saved to {RESULTS_FILE}")
    
    return results


logger.info("Starting TempReason dynamic query processing...")
loop = asyncio.get_event_loop()
query_results = loop.run_until_complete(process_timeqa_queries(graph_rag))

# Calculate average response time for successful queries
success_count = sum(1 for r in query_results if r["status"] == "success")
if success_count > 0:
    avg_query_time = sum(r.get("query_time", 0) for r in query_results if r["status"] == "success") / success_count
    logger.info(f"Average response time for successful TempReason queries: {avg_query_time:.2f} seconds")
    
    
################################################################################
# 5. Call evaluation module to evaluate results
################################################################################
try:
    from graphrag.evaluate import run_evaluation
    
    logger.info("Starting query result evaluation...")
    evaluation_metrics = run_evaluation(RESULTS_FILE)
    
    if evaluation_metrics:
        logger.info(f"Evaluation completed! F1: {evaluation_metrics['f1']:.2f}, ACC: {evaluation_metrics['accuracy']:.2f}")
except ImportError:
    logger.warning("Evaluation module not imported, skipping evaluation stage. Please use evaluate.py to evaluate results separately.")
except Exception as e:
    logger.error(f"Error in evaluation process: {str(e)}")
    import traceback
    traceback.print_exc()
