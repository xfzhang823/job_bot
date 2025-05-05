# your_tool.py
import logging
import logging_config
from local_llms.ds_distill_qwen1_5b.ds_model_loader import ModelLoader

from transformers import AutoModelForCausalLM

# AutoModelForCausalLM.from_pretrained(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", force_download=True
# )
tokenizer, model = ModelLoader.load_model()

# prompt = "Explain the concept of entropy in simple terms."
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.no_grad():
#     outputs = model.generate(**inputs, max_length=300)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
