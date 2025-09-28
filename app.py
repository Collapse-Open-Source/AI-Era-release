# app.py
import os
import re
import time
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Настройки ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "tinybert_model")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Папка 'tinybert_model' не найдена! Положите её рядом с app.py.")

# Устройство
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Загрузка модели
print("📥 Загружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("✅ Модель готова!")


# === Функции обработки ===
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('ё', 'е')
    replacements = {
        '0': 'о', '1': 'и', '3': 'з', '4': 'ч', '7': 'т',
        '@': 'а', '$': 'с', '+': 'т', '*': '', '#': '', '%': '',
        '!': '', '?': '', ',': '', ';': '', ':': '', '"': '', "'": '',
        '(': '', ')': '', '[': '', ']': '', '-': '', '_': '',
        'a': 'а', 'b': 'ь', 'c': 'с', 'd': "ь", 'e': "е", 'f': "ф", 'g': "д",
        'h': "х", 'i': "и", 'j': "и", 'k': "к", 'l': "л", 'm': "м", 'n': "п", 'o': "о", 'p': "р",
        'q': "д", 'r': "г", 's': "с", 't': "т", 'u': "и", 'v': "в", 'w': "ш", 'x': "х", 'y': "у", 'z': "з"
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^а-я\s]', ' ', text)
    return ' '.join(text.split())

    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^а-я\s]', ' ', text)
    return ' '.join(text.split())


def predict_mat(text):
    processed = preprocess_text(text)
    inputs = tokenizer(
        processed,
        truncation=True,
        padding="max_length",
        max_length=96,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE)
        )
        label = torch.argmax(outputs.logits, dim=1).cpu().item()
    return label == 1  # True если мат


# === Flask-приложение ===
app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"error": "Текст пустой"}), 400

    try:
        preprocessed_text = preprocess_text(text)
        is_mat = predict_mat(preprocessed_text)
        return jsonify({"result": 1 if is_mat else 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Сервер запущен: http://localhost:5001")
    app.run(host='0.0.0.0', port=5002, debug=False)