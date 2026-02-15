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
# [Critical Hotfix] 强制注册 Qwen3-VL 配置
# ------------------------------------------------------------------------------
# 解决报错: ValueError: Unrecognized configuration class ... Qwen3VLConfig
# 原因: Transformers 开发版虽然包含了 Qwen3 代码，但可能未正确注册到全局 AutoConfig 映射中。
# ==============================================================================
try:
    # 尝试直接从 transformers 内部导入 Qwen3 的相关类
    # 注意：如果你安装的是 transformers-main.zip，这些类应该存在
    try:
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLM
        
        # 1. 告诉 AutoConfig: "qwen3_vl" 这个字符串对应 Qwen2VLConfig (架构兼容)
        # 或者如果有 Qwen3VLConfig 就用 Qwen3 的
        try:
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
            AutoConfig.register("qwen3_vl", Qwen3VLConfig)
            print("✅ [Models] Successfully registered 'qwen3_vl' with Qwen3VLConfig.")
        except ImportError:
            # 如果还没发布 Qwen3VLConfig，则用 Qwen2VLConfig (完全兼容)
            AutoConfig.register("qwen3_vl", Qwen2VLConfig)
            print("⚠️ [Models] Qwen3VLConfig not found. Fallback: registered 'qwen3_vl' with Qwen2VLConfig.")

        # 2. 告诉 AutoModel: Qwen3VLConfig 对应的模型类是 Qwen2VLCausalLM (或者 Qwen3VLCausalLM)
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLM
            AutoModelForCausalLM.register(Qwen3VLConfig, Qwen3VLCausalLM)
        except (ImportError, NameError):
            AutoModelForCausalLM.register(Qwen2VLConfig, Qwen2VLCausalLM)
            
    except ImportError as e:
        print(f"⚠️ [Models] Registration Hotfix failed: {e}. Relying on AutoClasses default behavior.")

except Exception as e:
    print(f"⚠️ [Models] Unknown error during registration hotfix: {e}")
# ==============================================================================


class VerifierModel:
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        # 路径回退检查
        if not os.path.exists(model_name) and "models/" in model_name:
             # 如果本地没找到，尝试用 huggingface ID (用户可能没下载完)
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
        self.model.eval() # 确保进入推理模式

    def verify_claims(self, description):
        """
        Generates extraction of claims from the description.
        """
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
        
        # 清洗 DeepSeek 的思维链标签 (如果有)
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        
        claims = [line.strip().lstrip('-*').strip() for line in clean.split('\n') if len(line.strip()) > 5]
        return claims, raw

    def compute_sequence_log_prob(self, prompt, completion):
        """
        Computes the log probability of the completion given the prompt.
        """
        full_text = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:" + completion
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        # Mask out the prompt part for loss calculation
        prompt_text = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:"
        prompt_len = self.tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        
        # Ensure we don't mask everything if tokenization length varies slightly
        mask_len = min(prompt_len, labels.shape[1])
        labels[:, :mask_len] = -100
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                labels=labels
            )
        
        # Valid tokens for normalization
        valid_tokens = (labels != -100).sum()
        if valid_tokens == 0:
            return torch.tensor(0.0).to(self.device)
            
        return -outputs.loss * valid_tokens


class VLMModel:
    def __init__(self, model_name="./models/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        if not os.path.exists(model_name) and "models/" in model_name:
             print(f"⚠️ Local path {model_name} not found, trying HuggingFace ID...")
             model_name = "Qwen/Qwen2.5-VL-7B-Instruct" # Fallback if Qwen3 path is wrong
             
        print(f"Loading VLM: {model_name}")
        
        # Qwen-VL 系列通常需要 trust_remote_code=True 且 min_pixels/max_pixels 设置
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True,
                min_pixels=256*28*28, 
                max_pixels=1280*28*28
            )
        except Exception:
            # Fallback for standard loading
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
        """
        Generates descriptions for a batch of images using Qwen-VL prompt format.
        """
        # Qwen-VL 标准 Prompt 格式
        # 注意: 这里的实现假设 image_inputs 是 PIL Images 列表
        
        messages_batch = []
        for _ in image_inputs:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None}, # 占位符, processor 会处理
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                }
            ]
            messages_batch.append(messages)

        # 准备 inputs (Qwen2.5/3 VL 的 processor 处理方式)
        text_prompts = [
            self.processor.apply_chat_template(msg, add_generation_prompt=True) 
            for msg in messages_batch
        ]
        
        inputs = self.processor(
            text=text_prompts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=num_generations
            )

        # Decode output
        # Qwen generate 输出包含 input，需要截断
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids.repeat_interleave(num_generations, dim=0), generated_ids)
        ]
        
        texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        
        # Reshape: [batch_size, num_generations]
        reshaped_texts = [
            texts[i * num_generations : (i + 1) * num_generations] 
            for i in range(len(image_inputs))
        ]
        return reshaped_texts

    def compute_log_probs(self, input_ids, attention_mask, labels):
        """
        Wraps model forward pass for log probability computation.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
        return -outputs.loss * (labels != -100).sum()
