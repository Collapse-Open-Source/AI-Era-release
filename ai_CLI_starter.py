import os
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# We use tinybert.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "tinybert_model")

# Checking model existence
if not os.path.exists(MODEL_PATH):
    print("Directory 'moderation_model' not found, unfortunately...")
    print("First teach model!\nRun mod.py in order to teach a model.")
    exit(1)

# Device (no GPU support, because we couldn't process any data w/ GPU, cuz we have no laptops w/ a good GPU, which could provide us with computing resources)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("MPS is currently in use as a prefered device...")
else:
    DEVICE = torch.device("cpu")
    print("CPU is used as an available device...")
    print("!!! If it is possible, we recommend using MPS device (for example, Apple M3 CPU) in order to make model faster !!!")

# Загрузка модели
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Welcome! Enter some text in Russian\n")


async def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('ё', 'е')
    replacements = {
        '0': 'о', '1': 'и', '3': 'з', '4': 'ч', '7': 'т',
        '@': 'а', '$': 'с', '+': 'т', '*': '', '#': '', '%': '',
        '!': '', '?': '', ',': '', ';': '', ':': '', '"': '', "'": '',
        '(': '', ')': '', '[': '', ']': '', '-': '', '_': ''
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^а-я\s]', ' ', text)
    return ' '.join(text.split())


# Предсказание
async def predict_mat(text):
    processed = preprocess_text(text)
    inputs = tokenizer(
        processed,
        truncation=True,
        padding="max_length",
        max_length=96,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().item()
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000
    return prediction, inference_time_ms


if __name__ == "__main__":
    print("Running a CLI-based client of Exorcist")
    print("Enter text (or 'exit' in order to quit the application):\n")
    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in ("exit", "quit", "выход", ""):
                print("Bye!")
                break
            if not user_input:
                continue
            label, speed_ms = predict_mat(user_input)
            status = True if label == 1 else False
            print(f"Status (censored words existence): {status}")
        except KeyboardInterrupt:
            print("\nQuiting the application...")
            break
        except Exception as e:
            print(f"Error: {e}\n")
