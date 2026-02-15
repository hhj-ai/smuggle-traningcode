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
# [Critical Hotfix v4] Qwen3-VL 动态类伪装
# ------------------------------------------------------------------------------
# 解决报错: "The model class you are passing has a config_class attribute that is not consistent"
# 原理: 动态创建一个继承自 Qwen2VL 的新类，但强行将其 config_class 指向 Qwen3VLConfig
# ==============================================================================
try:
    print("🛠️  Applying Qwen3-VL Dynamic Class Hotfix...")

    # 1. 获取基座类 (Qwen2VL)
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    
    # 2. 获取目标配置 (Qwen3VLConfig)
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        TargetConfig = Qwen3VLConfig
        print("   -> Found native Qwen3VLConfig.")
    except ImportError:
        TargetConfig = Qwen2VLConfig
        print("   -> Qwen3VLConfig missing, falling back to Qwen2VLConfig.")

    # 3. 尝试获取原生 Qwen3 模型类，如果失败，则进行动态伪装
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        TargetModel = Qwen3VLForConditionalGeneration
        print("   -> Found native Qwen3VLForConditionalGeneration.")
    except ImportError:
        print("   ⚠️ Native Qwen3VL model class missing. Creating dynamic proxy...")
        
        # [核心魔法] 动态定义一个类，继承 Qwen2VL 的实现，但绑定 Qwen3VL 的配置
        class Qwen3VLForConditionalGeneration_Hotfix(Qwen2VLForConditionalGeneration):
            config_class = TargetConfig
            
        TargetModel = Qwen3VLForConditionalGeneration_Hotfix

    # 4. 执行注册 (AutoModelForCausalLM)
    # 因为我们修改了 config_class，这里的校验现在会通过
    try:
        AutoConfig.register("qwen3_vl", TargetConfig)
    except ValueError:
        pass # Ignore "already registered" error

    AutoModelForCausalLM.register(TargetConfig, TargetModel)
    AutoModel.register(TargetConfig, TargetModel)
    
    print(f"✅ [Models] Successfully mapped 'qwen3_vl' Config to {TargetModel.__name__}")

except Exception as e:
    print(f"❌ [Models] Registration failed: {e}")
    # 不退出，让后续代码碰运气

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
        if not os.path.exists(model_name) and "models/" in model_name:
             print(f"⚠️ Local path {model_name} not found, using fallback...")
        
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

        # 这里的 AutoModelForCausalLM 会自动使用我们上方注册的 Hotfix 类
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
