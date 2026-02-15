import torch
import re
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor, 
    AutoConfig,
    AutoModel
)

# ==============================================================================
# [Critical Hotfix] 强制注册 Qwen3-VL 配置 (修正类名版)
# ------------------------------------------------------------------------------
# 解决报错: Unrecognized configuration class ... Qwen3VLConfig
# 修正点: QwenVL 的模型类是 Qwen2VLForConditionalGeneration，不是 CausalLM
# ==============================================================================
try:
    print("🛠️  Applying Qwen3-VL Registration Hotfix...")
    
    # 1. 尝试导入 Qwen2VL 的配置和模型类 (作为 Qwen3 的替身)
    # 注意：Transformers 中 Qwen2VL 的类名是 Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    
    # 2. 尝试导入 Qwen3VL (如果存在)
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        TargetConfig = Qwen3VLConfig
    except ImportError:
        TargetConfig = Qwen2VLConfig # Fallback
        
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        TargetModel = Qwen3VLForConditionalGeneration
    except ImportError:
        TargetModel = Qwen2VLForConditionalGeneration # Fallback

    # 3. 注册到 AutoConfig (解决 "model_type": "qwen3_vl" 无法识别的问题)
    AutoConfig.register("qwen3_vl", TargetConfig)
    
    # 4. 注册到 AutoModelForCausalLM (解决 Unrecognized configuration class)
    # 关键：告诉 AutoModelForCausalLM，遇到这个 Config 时，加载 TargetModel 类
    AutoModelForCausalLM.register(TargetConfig, TargetModel)
    AutoModel.register(TargetConfig, TargetModel)
    
    print(f"✅ [Models] Successfully mapped 'qwen3_vl' to {TargetModel.__name__}")

except Exception as e:
    print(f"⚠️ [Models] Registration Hotfix failed: {e}")
    print("   -> Attempting to proceed, but load might fail.")

# ==============================================================================

class VerifierModel:
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        if not os.path.exists(model_name) and "models/" in model_name:
             print(f"⚠️ Local path {model_name} not found, trying HuggingFace ID...")
             model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        
        print(f"Loading Verifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map={"": device}, 
            trust_remote_code=True, 
            attn_implementation="flash_attention_2"
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
        # 路径回退
        if not os.path.exists(model_name) and "models/" in model_name:
             print(f"⚠️ Local path {model_name} not found, checking fallback...")
             # 如果实在没有，可以 fallback 到 Qwen2-VL
        
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

        # 关键修改：直接使用 AutoModelForCausalLM 加载，依赖上方的 Hotfix
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
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
