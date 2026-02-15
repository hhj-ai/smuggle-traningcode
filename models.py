import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

# ==============================================================================
# 🔧 核心修改：动态注册 Qwen3 架构别名
# ==============================================================================
def register_custom_architectures():
    """
    在内存中将 'qwen3_vl' 注册为 'Qwen2VL' 的子类/别名。
    这样无需修改 config.json 文件，transformers 也能正确识别架构。
    """
    try:
        # 尝试导入 Qwen2VL 的配置和模型类（需要 transformers >= 4.45.0）
        from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
        
        print("🛠️  正在执行架构注册: Mapping 'qwen3_vl' -> Qwen2VL classes...")
        
        # 1. 注册配置类：告诉 AutoConfig 遇到 "qwen3_vl" 时使用 Qwen2VLConfig
        AutoConfig.register("qwen3_vl", Qwen2VLConfig)
        
        # 2. 注册模型类：告诉 AutoModel 遇到这个配置时加载哪个模型类
        AutoModelForCausalLM.register(Qwen2VLConfig, Qwen2VLForConditionalGeneration)
        
        print("✅  架构注册成功！现在可以直接加载 Qwen3-VL 了。")
        
    except ImportError:
        print("\n⚠️  [严重警告] 你的 transformers 版本过低，无法导入 Qwen2VL 基类！")
        print("   这会导致 Qwen3-VL 加载失败。请务必运行: pip install --upgrade transformers\n")
    except Exception as e:
        print(f"⚠️  架构注册过程中出现非致命错误: {e}")

# 在模块导入时立即执行注册
register_custom_architectures()
# ==============================================================================

class VerifierModel:
    """
    Wrapper for Verifier.
    Defaults to local path './models/DeepSeek-R1-Distill-Qwen-7B'.
    """
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        
        # 路径检查
        if not os.path.exists(model_name):
            print(f"⚠️ Warning: Local model path '{model_name}' not found. Fallback to HF ID.")
            if "models/" in model_name: 
                model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        
        print(f"Loading Verifier from: {model_name} ...")
        
        # DeepSeek R1 使用的是标准的 Llama/Qwen 结构，通常不需要特殊注册
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device}, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def verify_claims(self, description):
        prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {description}\n\nClaims:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6
        )
        
        raw_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        claims = []
        for line in clean_text.split('\n'):
            cleaned = line.strip().lstrip('-').lstrip('*').strip()
            if len(cleaned) > 5:
                claims.append(cleaned)
                
        return claims, raw_response

    def compute_sequence_log_prob(self, prompt, completion):
        full_prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:"
        full_text = full_prompt + completion
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        prompt_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        safe_len = min(prompt_len, labels.shape[1])
        labels[:, :safe_len] = -100
        
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
        
        valid_token_count = (labels != -100).sum().item()
        if valid_token_count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        total_log_prob = -outputs.loss * valid_token_count
        return total_log_prob

class VLMModel:
    """
    Wrapper for VLM.
    Defaults to local path './models/Qwen3-VL-8B-Instruct'.
    """
    def __init__(self, model_name="./models/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        
        if not os.path.exists(model_name):
            print(f"⚠️ Warning: Local model path '{model_name}' not found. Fallback to HF ID.")
            if "models/" in model_name:
                model_name = "Qwen/Qwen3-VL-8B-Instruct"

        print(f"Loading VLM from: {model_name} ...")
        
        try:
            # 这里的 Processor 加载通常依赖 qwen2_vl 的处理逻辑
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # 由于我们在文件头部做了 register_custom_architectures()
            # 这里 AutoModel 应该能自动识别 qwen3_vl 并调用 Qwen2VL 类
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
        except Exception as e:
            print(f"❌ VLM Load Error Details: {e}")
            raise RuntimeError(f"VLM Load Error: {e}")
            
        self.tokenizer = self.processor.tokenizer

    def generate_description_batch(self, image_inputs, num_generations=4):
        # Qwen2/3-VL 的标准 Prompt 格式
        text_prompts = ["Describe this image in detail."] * len(image_inputs)
        
        inputs = self.processor(
            text=text_prompts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=num_generations
            )
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        results = []
        for i in range(len(image_inputs)):
            start = i * num_generations
            results.append(generated_texts[start : start + num_generations])
            
        return results

    def compute_log_probs(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        valid_count = (labels != -100).sum().item()
        if valid_count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return -outputs.loss * valid_count
