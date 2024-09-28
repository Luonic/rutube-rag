from transformers import AutoModel
from peft import get_peft_model, LoraConfig

model = AutoModel.from_pretrained("deepvk/USER-bge-m3")
print(type(model))
print(model.config)
peft_model = get_peft_model(model, peft_config=LoraConfig())
print(type(peft_model))
model = peft_model.merge_and_unload()
print(type(model))
print(model.config)