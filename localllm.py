import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tenacity import retry, stop_after_attempt, wait_random_exponential


    
class LlamaQAModel:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        attn_implementation = "flash_attention_2" if _is_flash_attn_available() else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation
        )

        self.device = next(self.model.parameters()).device

    def generate(self, input_text, **kwargs):
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample =kwargs.get("do_sample", False),
            # temperature=kwargs.get("temperature", 0.1),
            num_beams=kwargs.get("num_beams", 1),
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question):
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional research assistant. Please answer the questions based strictly on the provided information.\n"
                        # "Note:\n"
                        # "1. If the answer can be clearly inferred from the provided information (e.g., based on time periods or events mentioned), you should present it directly.\n"
                        # "2. Do not make up or guess answers beyond what is supported by the information.\n"
                        # "3. If the information is insufficient or does not contain any relevant facts to answer the question, you should say N/A.\n"
                        # "4. When information about a specific time point is provided (e.g., 1900) and a subsequent change is mentioned at a later date (e.g., 1902), you can infer that the situation remained unchanged during the intervening period unless stated otherwise.\n"
                        # "5. Identify the exact entity (person, organization, location, etc.) that most directly and unambiguously answers the query. Output only this entity's name.\n"
                        "Reference Contexts:\n"
                        f"{context}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Based on the information provided, " + question + " Output only the most direct single entity(1-4 words). If no answer, output \"N/A\".\n"
                }
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            response = self.generate(input_text)
            return response
        except Exception as e:
            print(f"[QA Error] {e}")
            raise  # 保留 retry 机制


# ===== 辅助函数：安全检测 FlashAttention =====
def _is_flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False