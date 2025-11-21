import torch
import gc
import json
import traceback
import sys
import datetime
import logging
import os
import re
import string
import asyncio
import time
import shutil
import tempfile
import aiohttp
from collections import Counter
from typing import List
from pathlib import Path

# ====================== 超参数配置 ======================
DATASET_FILE = "/root/raptor/test.hard.json"  # 数据集文件路径
QUERY_MODE = "dynamic"
QUERY_TOP_K = 20
CONCURRENCY = 5  # 控制并发请求
LLM_MODEL_NAME = "Llama3.1-8B-Instruct"  # GraphRAG内部使用的LLM模型
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
# ======================================================

# GraphRAG 依赖
from graphrag import GraphRAG, QueryParam
from graphrag._utils import compute_args_hash

# 统一的生成器模型
from localllm import LlamaQAModel
logger = logging.getLogger("GraphRAG-Eval")

# ################################################################################
# # 0. 启动vllm服务
# ################################################################################
# # 在脚本开头添加
# try:
#     from vllm_manager import vllm_manager
#     if not vllm_manager.start():
#         logger.error("[x] Failed to start vLLM server")
#         sys.exit(1)
# except Exception as e:
#     logger.error(f"[x] VLLM manager error: {e}")
#     sys.exit(1)

# 修改VLLM_BASE_URL
VLLM_BASE_URL = "http://localhost:8001/v1"


################################################################################
# 1. 配置加载
################################################################################
print("🔧 Loading configuration...")

LOCAL_BGE_PATH = EMBEDDING_MODEL_NAME
OPENAI_API_KEY_FAKE = "EMPTY"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

for module in ['tzlocal', 'urllib3', 'faiss', 'transformers', 'sentence_transformers']:
    logging.getLogger(module).setLevel(logging.WARNING)

################################################################################
# 2. 评估指标
################################################################################
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text, flags=re.IGNORECASE)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()

def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f1

def check_containment_relationship(gold: str, pred: str) -> bool:
    """检查是否满足包含关系：gold完全包含pred或pred完全包含gold"""
    if gold == "N/A":
        return pred == "N/A"
    
    if pred == "N/A" and gold != "N/A":
        return False
    
    norm_gold = normalize_answer(gold)
    norm_pred = normalize_answer(pred)
    
    if not norm_gold or not norm_pred:
        return False
        
    return norm_gold in norm_pred or norm_pred in norm_gold

################################################################################
# 3. 按需图谱构建 GraphRAG
################################################################################
class OnDemandGraphRAGRetriever:
    """按需构建图谱的 GraphRAG 检索器，每次查询都从头构建新图谱"""
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(CONCURRENCY)
        self.timeout_exceptions = (
            asyncio.TimeoutError,
            aiohttp.client_exceptions.ClientConnectorError,
            aiohttp.client_exceptions.ServerTimeoutError,
            aiohttp.client_exceptions.ClientOSError,
            ConnectionRefusedError,
            ConnectionError,
            TimeoutError,
        )
        self.max_retries = 3
        self.retry_delay = 10  # 秒
    
    async def _create_graphrag_instance(self, working_dir: Path) -> GraphRAG:
        """创建 GraphRAG 实例，使用临时工作目录"""
        from dataclasses import dataclass
        from sentence_transformers import SentenceTransformer
        from openai import AsyncOpenAI
        import torch
        
        # 嵌入函数
        @dataclass
        class EmbeddingFunc:
            embedding_dim: int
            max_token_size: int
            model: SentenceTransformer = None
            
            async def __call__(self, texts: List[str]) -> list:
                if isinstance(texts, str):
                    texts = [texts]
                loop = asyncio.get_running_loop()
                # 使用同步方式调用模型，避免嵌套异步
                return await loop.run_in_executor(None, lambda: self.model.encode(
                    texts,
                    batch_size=8,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ))
        
        # 初始化嵌入模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st_model = SentenceTransformer(
            LOCAL_BGE_PATH,
            device=device,
            trust_remote_code=True
        )
        
        embedding_func = EmbeddingFunc(
            embedding_dim=st_model.get_sentence_embedding_dimension(),
            max_token_size=8192,
            model=st_model
        )
        
        # Llama3.1-8B-Instruct 的异步调用函数
        def _build_async_client():
            return AsyncOpenAI(api_key=OPENAI_API_KEY_FAKE, base_url=VLLM_BASE_URL)
        
        async def llama_model_func(prompt: str, system_prompt: str | None = None, **kwargs) -> str:
            """Llama3.1-8B-Instruct 的调用函数"""
            # 从kwargs中移除GraphRAG特定的参数
            hashing_kv = kwargs.pop("hashing_kv", None)  # 移除hashing_kv参数，不传递给API
            
            client = _build_async_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                logger.error(traceback.format_exc())
                return f"ERROR_LLM_{str(e)[:50]}"
        
        # 创建 GraphRAG 实例，使用临时工作目录
        return GraphRAG(
            working_dir=str(working_dir),
            embedding_func=embedding_func,
            best_model_func=llama_model_func,
            cheap_model_func=llama_model_func,
            enable_llm_cache=False,
            best_model_max_token_size=8192,
            cheap_model_max_token_size=8192,
        )
    
    async def retrieve_context_from_single_document(self, question: str, document: str) -> str:
        """为单个文档构建图谱并检索上下文"""
        # 创建临时工作目录
        temp_dir = Path(tempfile.mkdtemp(prefix="graphrag_eval_"))
        logger.debug(f"Creating temporary GraphRAG directory: {temp_dir}")
        
        try:
            # 1. 创建 GraphRAG 实例
            graph_rag = await self._create_graphrag_instance(temp_dir)
            
            # 2. 插入文档并构建图谱
            start_build = time.time()
            await graph_rag.ainsert(document)
            build_time = time.time() - start_build
            logger.debug(f"Graph built in {build_time:.2f} seconds for question: {question[:30]}...")
            
            # 3. 检索相关上下文
            query_param = QueryParam(
                mode=QUERY_MODE,
                top_k=QUERY_TOP_K,
                only_need_context=True,  # 只返回检索到的上下文
                response_type="Short Answer or Direct Response"
            )
            
            start_query = time.time()
            context = await graph_rag.aquery(question, param=query_param)
            query_time = time.time() - start_query
            logger.debug(f"Context retrieved in {query_time:.2f} seconds")
            
            # 4. 返回上下文
            return context
            
        finally:
            # 5. 清理临时目录（确保即使出错也会清理）
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

    async def retrieve_context(self, question: str, context: str) -> str:
        """检索相关上下文，不生成答案 (兼容接口)"""
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    start_time = time.time()
                    retrieved_context = await self.retrieve_context_from_single_document(question, context)
                    total_time = time.time() - start_time
                    logger.debug(f"Full process completed in {total_time:.2f}s for: {question[:50]}...")
                    return retrieved_context
            
            except self.timeout_exceptions as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for question '{question[:20]}...': {str(e)}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"Context retrieval failed after {self.max_retries} attempts: {question}")
                    return f"ERROR_TIMEOUT_{str(e)}"
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                return f"ERROR_{str(e)[:50]}"

################################################################################
# 4. 评估主函数
################################################################################
async def evaluate_dataset() -> bool:
    """评估 GraphRAG 检索 + 统一生成器在 TimeQA 上的表现"""
    # 初始化组件
    retriever = OnDemandGraphRAGRetriever()
    generator = LlamaQAModel()  # 统一的生成器模型
    
    # 配置日志
    base_name = os.path.splitext(os.path.basename(DATASET_FILE))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_ondemand_graphrag_{timestamp}_log.txt"
    results_file = f"{base_name}_ondemand_graphrag_{timestamp}_results.json"
    
    # 重配置文件日志
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting on-demand evaluation: GraphRAG builds fresh graph for each question")
    logger.info(f"GraphRAG Internal LLM: {LLM_MODEL_NAME}")
    logger.info(f"Configuration: mode={QUERY_MODE}, top_k={QUERY_TOP_K}, concurrency={CONCURRENCY}")
    logger.info(f"Results will be saved to: {results_file}")
    
    # 初始化指标
    em_total = 0.0
    f1_total = 0.0
    corrected_em_total = 0.0
    corrected_f1_total = 0.0
    total = 0
    interrupted = False
    start_time = time.time()
    results = []
    
    try:
        # 加载数据集
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        logger.info(f"📚 Loaded {len(data)} questions from {DATASET_FILE}")
        
        # 处理每个问题
        for idx, entry in enumerate(data):
            try:
                question_id = entry.get("id", f"q{idx}")
                question = entry.get("question", "")
                targets = entry.get("targets", [])
                context = entry.get("context", "") 
                
                # 如果有多个上下文，合并为一个（按需图谱构建通常处理单文档）
                if isinstance(context, list):
                    document = "\n\n".join(context)
                else:
                    document = context
                
                if not document.strip():
                    logger.warning(f"⚠️  Empty context for question {question_id}, skipping")
                    continue
                
                logger.info(f"🔍 Processing question {idx+1}/{len(data)}: ID={question_id}, Text='{question[:30]}...'")
                
                # 1. 检索上下文 (这会构建新图谱)
                retrieved_context = await retriever.retrieve_context(question, document)
                
                # 2. 使用统一生成器生成答案
                if retrieved_context.startswith("ERROR_"):
                    prediction = "ERROR_GENERATION"
                else:
                    try:
                        prediction = generator.answer_question(retrieved_context, question)
                    except Exception as e:
                        logger.error(f"GenerationStrategy error for {question_id}: {str(e)}")
                        prediction = f"ERROR_{str(e)[:50]}"
                
                # 3. 处理目标答案
                processed_targets = ["N/A" if t.strip() == "" else t for t in targets] or ["N/A"]
                
                # 4. 计算原始分数
                em_scores = [compute_exact(t, prediction) for t in processed_targets]
                f1_scores = [compute_f1(t, prediction) for t in processed_targets]
                em_score = max(em_scores) if em_scores else 0.0
                f1_score = max(f1_scores) if f1_scores else 0.0
                
                # 5. 计算修正分数
                corrected_em_score = em_score
                corrected_f1_score = f1_score
                containment_applied = False
                
                if em_score < 1.0:
                    for target in processed_targets:
                        if check_containment_relationship(target, prediction):
                            corrected_em_score = 1.0
                            corrected_f1_score = 1.0
                            containment_applied = True
                            break
                
                # 6. 累加分数
                em_total += em_score
                f1_total += f1_score
                corrected_em_total += corrected_em_score
                corrected_f1_total += corrected_f1_score
                total += 1
                
                # 7. 保存结果
                result = {
                    "question_id": question_id,
                    "question": question,
                    "original_context": document[:200] + "..." if len(document) > 200 else document,
                    "retrieved_context": retrieved_context[:300] + "..." if len(retrieved_context) > 300 else retrieved_context,
                    "prediction": prediction,
                    "targets": processed_targets,
                    "original_em": em_score,
                    "original_f1": f1_score,
                    "corrected_em": corrected_em_score,
                    "corrected_f1": corrected_f1_score,
                    "containment_applied": containment_applied,
                    "error": retrieved_context.startswith("ERROR_") or prediction.startswith("ERROR_")
                }
                results.append(result)
                
                # 8. 日志记录
                status = "✅" if em_score == 1 else "❌"
                containment_note = " (CONTAINMENT CORRECTED)" if containment_applied else ""
                log_entry = (
                    f"\n{'='*60}\n"
                    f"{status} ID: {question_id}{containment_note}\n"
                    f"Question: {question}\n"
                    f"Targets: {processed_targets}\n"
                    f"Prediction: {prediction}\n"
                    f"Original Scores - EM: {em_score:.0f}, F1: {f1_score:.4f}\n"
                    f"Corrected Scores - EM: {corrected_em_score:.0f}, F1: {corrected_f1_score:.4f}\n"
                    f"{'='*60}\n"
                )
                logger.info(log_entry)
                
                # 每5个问题保存中间结果
                if (idx + 1) % 5 == 0:
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "metadata": {
                                "dataset": DATASET_FILE,
                                "total_processed": total,
                                "timestamp": datetime.datetime.now().isoformat(),
                                "config": {
                                    "mode": QUERY_MODE,
                                    "top_k": QUERY_TOP_K,
                                    "generator_model": generator.__class__.__name__,
                                    "internal_llm": LLM_MODEL_NAME,
                                    "evaluation_type": "on_demand_per_document"
                                }
                            },
                            "results": results
                        }, f, indent=2, ensure_ascii=False)
                    logger.info(f"💾 Saved intermediate results after {idx+1} questions")
                    
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("🧹 GPU memory cache cleared after QA step")
            
            except KeyboardInterrupt:
                logger.warning(f"⚠️ Evaluation interrupted by user at question {idx}")
                interrupted = True
                break
            except Exception as e:
                logger.error(f"🔥 Critical error at question {idx}: {str(e)}")
                traceback.print_exc()
                continue
    
    except Exception as e:
        logger.error(f"🔥 Critical error loading dataset: {str(e)}")
        traceback.print_exc()
    
    finally:
        # 计算最终指标
        if total > 0:
            em_avg = (em_total / total) * 100
            f1_avg = (f1_total / total) * 100
            corrected_em_avg = (corrected_em_total / total) * 100
            corrected_f1_avg = (corrected_f1_total / total) * 100
            
            # 保存最终结果
            final_results = {
                "metadata": {
                    "dataset": DATASET_FILE,
                    "total_questions": total,
                    "query_mode": QUERY_MODE,
                    "top_k": QUERY_TOP_K,
                    "generator_model": generator.__class__.__name__,
                    "internal_llm": LLM_MODEL_NAME,
                    "evaluation_type": "on_demand_per_document",
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.datetime.now().isoformat()
                },
                "metrics": {
                    "original": {
                        "em": f"{em_avg:.2f}%",
                        "f1": f"{f1_avg:.2f}%"
                    },
                    "corrected": {
                        "em": f"{corrected_em_avg:.2f}%",
                        "f1": f"{corrected_f1_avg:.2f}%"
                    },
                    "improvement": {
                        "em_delta": f"{corrected_em_avg - em_avg:.2f}%",
                        "f1_delta": f"{corrected_f1_avg - f1_avg:.2f}%"
                    }
                },
                "results": results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            # 打印总结
            summary = (
                f"\n{'='*60}\n"
                f"[{'INTERRUPTED' if interrupted else 'COMPLETED'}] Evaluation Results\n"
                f"Total Questions Processed: {total}\n"
                f"Time Elapsed: {time.time() - start_time:.1f} seconds\n"
                f"{'-'*60}\n"
                f"Original Scores:\n"
                f"  EM: {em_avg:.2f}%\n"
                f"  F1: {f1_avg:.2f}%\n"
                f"Corrected Scores (with containment):\n"
                f"  EM: {corrected_em_avg:.2f}%\n"
                f"  F1: {corrected_f1_avg:.2f}%\n"
                f"Improvement:\n"
                f"  EM: +{corrected_em_avg - em_avg:.2f}%\n"
                f"  F1: +{corrected_f1_avg - f1_avg:.2f}%\n"
                f"{'='*60}\n"
                f"Results saved to: {results_file}\n"
            )
            logger.info(summary)
            print(summary)
        else:
            logger.error(" No questions were successfully processed!")
    
    return interrupted

################################################################################
# 5. 主入口
################################################################################
if __name__ == "__main__":
    try:
        print(f"Starting on-demand GraphRAG evaluation (building fresh graph for each question)")
        print(f"Dataset: {DATASET_FILE}")
        print(f"GraphRAG Internal LLM: {LLM_MODEL_NAME}")
        print(f"Configuration: mode={QUERY_MODE}, top_k={QUERY_TOP_K}")
        interrupted = asyncio.run(evaluate_dataset())
        exit_code = 1 if interrupted else 0
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        traceback.print_exc()
        exit_code = 2
    
    sys.exit(exit_code)