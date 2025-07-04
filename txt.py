from transformers import pipeline
import os


os.environ["HF_HUB_ENABLE_XET"] = "0"


generator = pipeline("text-generation", model="distilgpt2")



prompt = "Artificial intelligence is future of"


output = generator(prompt, max_length=50, num_return_sequences=1)


print("Generated Text:\n")
print(output[0]['generated_text'])