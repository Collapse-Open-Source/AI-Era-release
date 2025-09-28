import os
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 Устройство: {DEVICE}")
MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LEN = 96
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "tinybert_model")


def read_csv_skip_first(filepath):
    for encoding in ['utf-8', 'utf-8-sig']:
        for sep in [',', ';']:
            try:
                df = pd.read_csv(filepath, sep=sep, header=None, encoding=encoding, skiprows=1)
                if len(df.columns) >= 2:
                    return df
            except:
                continue
    raise ValueError("Couldn't read the file")


def parse_label(lbl):
    if isinstance(lbl, (int, float)):
        val = int(lbl)
        return val if val in (0, 1) else None
    if isinstance(lbl, str):
        clean = lbl.strip().lower()
        if clean in {'0', '0.0', 'false', 'no'}:
            return 0
        if clean in {'1', '1.0', 'true', 'yes'}:
            return 1
    return None


class ModerationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def train_model(filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(filepath):
        print(f"Couldn't find the object with following name'{filename}'")
        return False

    df = read_csv_skip_first(filepath)
    if df.shape[1] < 3:
        print("3 Columns are needed")
        return False

    texts = df.iloc[:, 1].fillna("").tolist()
    labels_raw = df.iloc[:, 2].tolist()
    labels = []
    valid_texts = []
    for i, lbl in enumerate(labels_raw):
        parsed = parse_label(lbl)
        if parsed is not None:
            labels.append(parsed)
            valid_texts.append(texts[i])

    if len(set(labels)) < 2:
        print("Both examples are needed (0 and 1)")
        return False

    print(f"Data: {len(labels)} lines; Classes: {labels.count(0)} / {labels.count(1)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    dataset = ModerationDataset(valid_texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        total_loss = 0
        for batch in tqdm(dataloader, desc="Teaching", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_batch = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Average loss: {avg_loss:.4f}")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model has been saved to following path: {MODEL_SAVE_PATH}")
    return True


def predict(filename, has_labels=False, output_name="predictions.csv"):
    filepath = os.path.join(SCRIPT_DIR, filename)
    if not os.path.isfile(filepath):
        print(f"Couldn't find an object with following name: '{filename}'")
        return None

    if not os.path.exists(MODEL_SAVE_PATH):
        print("First teach the model")
        return None

    df = read_csv_skip_first(filepath)
    if has_labels and df.shape[1] < 3:
        print("3 Columns are needed for check")
        return None
    if not has_labels and df.shape[1] < 2:
        print("2 Columns are needed")
        return None

    ids = df.iloc[:, 0].tolist()
    texts = df.iloc[:, 1].fillna("").tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.to(DEVICE)
    model.eval()

    predictions = []
    for text in tqdm(texts, desc="Predicting", leave=False):
        enc = tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).cpu().item()
            predictions.append(pred)

    pd.DataFrame({"id": ids, "label": predictions}).to_csv(
        os.path.join(SCRIPT_DIR, output_name),
        index=False,
        header=False
    )
    print(f"Result: {output_name}")

    if has_labels:
        true_labels = []
        for lbl in df.iloc[:, 2]:
            parsed = parse_label(lbl)
            true_labels.append(parsed if parsed is not None else 0)
        true_labels = true_labels[:len(predictions)]
        if len(true_labels) == len(predictions):
            accuracy = accuracy_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions, average="macro")
            f1 = f1_score(true_labels, predictions, average="binary")
            print(f" F1-score: {f1:.4f}, accuracy: {accuracy:.4f}, recall: {recall:.4f}")
    return predictions


def main():
    print("\nUsed Rubert Tiny 2")
    print("Teach time - approximately 90 minutes")
    print("format: first line in CSV is ignored, then it requires following columns: id,text,label\n")

    print("1. Teach")
    print("2. Reteach")
    print("3. Check (F1, Average, Recall)")
    print("4. Predict")

    choice = input("\nChoose an option: ").strip()

    if choice in ("1", "2"):
        fn = input("Name of the file, on which you wanna teach the model (e.g. input.csv): ").strip()
        train_model(fn)

    elif choice == "3":
        fn = input("Name of the files, on which you wanna check score (e.g. input.csv): ").strip()
        predict(fn, has_labels=True, output_name="f1_tiny.csv")

    elif choice == "4":
        fn = input("File name (e.g. input.csv): ").strip()
        out = input("Result file name (e.g. out.csv): ").strip() or "pred_tiny.csv"
        predict(fn, has_labels=False, output_name=out)

    else:
        print("Unrecognized option.")


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        print("MPS is unavailable. Using CPU instead.")
    main()