import torch
import re
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor, 
    AutoConfig
)

# ==============================================================================
# [Strategy] Direct Class Loading
# ------------------------------------------------------------------------------
# 既然原生类存在，直接引用类进行加载，绕过 AutoModel 的配置映射地狱。
# ==============================================================================
Qwen3VLClass = None
try:
    # 尝试直接导入 Qwen3VL 的模型类
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    Qwen3VLClass = Qwen3VLForConditionalGeneration
    print("✅ [Models] Using native Qwen3VLForConditionalGeneration for loading.")
except ImportError:
    print("⚠️ [Models] Qwen3VL native class not found. Trying Qwen2VL fallback...")
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        Qwen3VLClass = Qwen2VLForConditionalGeneration
    except ImportError:
        print("❌ [Models] Critical: No compatible QwenVL model class found!")

# ==============================================================================

class VerifierModel:
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        if not os.path.exists(model_name) and "models/" in model_name:
             model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        
        print(f"Loading Verifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Verifier 通常是标准语言模型，AutoModel 没问题
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map={"": device}, 
            trust_remote_code=True, 
            attn_implementation="sdpa",
            local_files_only=True
        )
        self.model.eval()

    def verify_claims(self, description):
        prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {description}\n\nClaims:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.6,
                pad_token_id=self.tokenizer.pad_token_id
            )
        raw = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        claims = [line.strip().lstrip('-*').strip() for line in clean.split('\n') if len(line.strip()) > 5]
        return claims, raw

    def compute_sequence_log_prob(self, prompt, completion):
        full_text = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:" + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        prompt_text = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:"
        prompt_len = self.tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        labels[:, :min(prompt_len, labels.shape[1])] = -100
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            
        valid_tokens = (labels != -100).sum()
        if valid_tokens == 0: return torch.tensor(0.0).to(self.device)
        return -outputs.loss * valid_tokens


class VLMModel:
    def __init__(self, model_name="./models/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        if not os.path.exists(model_name) and "models/" in model_name:
             print(f"⚠️ Local path {model_name} not found.")
        
        print(f"Loading VLM: {model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True,
                min_pixels=256*28*28, 
                max_pixels=1280*28*28
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # 核心修改：使用具体类 Qwen3VLClass 而不是 AutoModelForCausalLM
        if Qwen3VLClass:
            print(f"🔹 Using direct class loading: {Qwen3VLClass.__name__}")
            self.model = Qwen3VLClass.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
                attn_implementation="sdpa",
            local_files_only=True
            )
        else:
            # 最后的兜底
            print("🔸 Fallback to AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
                attn_implementation="sdpa",
            local_files_only=True
            )
            
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

    def generate_description_batch(self, image_inputs, num_generations=4):
        messages_batch = []
        for _ in image_inputs:
            messages = [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": "Describe this image in detail."}]}]
            messages_batch.append(messages)

        text_prompts = [self.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in messages_batch]
        inputs = self.processor(text=text_prompts, images=image_inputs, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=1.0, num_return_sequences=num_generations)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids.repeat_interleave(num_generations, dim=0), generated_ids)]
        texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return [texts[i * num_generations : (i + 1) * num_generations] for i in range(len(image_inputs))]

    def compute_log_probs(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return -outputs.loss * (labels != -100).sum()
