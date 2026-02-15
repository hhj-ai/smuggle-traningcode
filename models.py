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
# [Critical Hotfix v3] 强制模型映射 (逻辑分离版)
# ------------------------------------------------------------------------------
# 修复: 即使 AutoConfig 报错，也要强制执行 AutoModel 的注册
# ==============================================================================

# 1. 准备目标类 (Qwen2VL 模型类是通用的)
try:
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    print("✅ [Models] Loaded Qwen2VL classes for mapping.")
except ImportError as e:
    print(f"❌ [Models] Critical: Qwen2VL classes missing. Update transformers! {e}")
    # 这里的 fallback 只是为了防崩，实际上如果缺这个后面大概率跑不了
    Qwen2VLConfig = None
    Qwen2VLForConditionalGeneration = None

# 2. 尝试获取 Qwen3VL 配置类 (如果源码里有)
TargetConfig = None
try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    TargetConfig = Qwen3VLConfig
    print("✅ [Models] Found native Qwen3VLConfig.")
except ImportError:
    TargetConfig = Qwen2VLConfig
    print("⚠️ [Models] Qwen3VLConfig not found. Using Qwen2VLConfig as proxy.")

# 3. [Step A] 注册 Config (允许失败)
if TargetConfig:
    try:
        # 尝试将 "qwen3_vl" 字符串绑定到配置类
        AutoConfig.register("qwen3_vl", TargetConfig)
    except ValueError:
        # 如果报错 "already used"，说明官方已经注册了，这是好事，直接跳过
        print("ℹ️  [Models] 'qwen3_vl' config already registered. Skipping.")
    except Exception as e:
        print(f"⚠️ [Models] Config registration warning: {e}")

# 4. [Step B] 注册 Model (关键步骤 - 必须执行!)
if TargetConfig and Qwen2VLForConditionalGeneration:
    try:
        # 强制告诉 AutoModel: 看到这个 Config，就用 Qwen2VLForConditionalGeneration 加载
        # 即使 Config 是 Qwen3VLConfig，因为架构相同，用 Qwen2VL 的模型代码也是兼容的
        AutoModelForCausalLM.register(TargetConfig, Qwen2VLForConditionalGeneration)
        AutoModel.register(TargetConfig, Qwen2VLForConditionalGeneration)
        print(f"✅ [Models] Force-registered {TargetConfig.__name__} -> Qwen2VLForConditionalGeneration")
    except Exception as e:
        print(f"❌ [Models] Model registration failed: {e}")

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

        # 这里 AutoModelForCausalLM 会利用我们在文件头注册的映射关系
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
        # Reshape
        return [texts[i * num_generations : (i + 1) * num_generations] for i in range(len(image_inputs))]

    def compute_log_probs(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return -outputs.loss * (labels != -100).sum()
