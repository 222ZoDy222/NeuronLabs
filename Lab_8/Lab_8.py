import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download  # можно оставить, даже если не используете ниже


model_name = "Qwen/Qwen2.5-7B-Instruct-1M"



model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"   
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


file_path = "ENG_article.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

print("Chars:", len(data))
print("Preview:", data[:200].replace("\n", " "))


questions = [
    "В каком году была обозначена проблема взрывающихся градиентов?",
    "Кто в 1891 году разработал метод уничтожающей производной?",
    "Кто предложил цепное правило дифференцирования и в каком году?"
]


system_prompt = (
    "Ты — ассистент для извлечения фактов из текста. "
    "Отвечай строго по вопросам. "
    "Если возможно, после каждого ответа дай короткий фрагмент из текста (на английском) как подтверждение."
)

user_prompt = (
    "Проанализируй следующий текст (английский). Найди в нём ответы на вопросы.\n\n"
    "ВОПРОСЫ:\n"
    "1) В каком году была обозначена проблема взрывающихся градиентов?\n"
    "2) Кто в 1891 году разработал метод уничтожающей производной?\n"
    "3) Кто предложил цепное правило дифференцирования и в каком году?\n\n"
    "ФОРМАТ ОТВЕТА (строго):\n"
    "1) <год> — <кратко> [доказательство: \"...\"]\n"
    "2) <имя> — <кратко> [доказательство: \"...\"]\n"
    "3) <имя>, <год> — <кратко> [доказательство: \"...\"]\n\n"
    "ТЕКСТ:\n"
    f"{data}"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.2,          
    do_sample=True
)

generated_ids_ = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids_, skip_special_tokens=True)[0]
print("\n=== ANSWER ===\n")
print(response)
