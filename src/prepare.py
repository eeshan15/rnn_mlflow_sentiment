import pandas as pd
import torch
import os
import json


os.makedirs("data/processed", exist_ok=True)


df = pd.read_csv("data/raw_data.csv")

all_text = " ".join(df['text'].tolist())
words = set(all_text.split())
vocab = {word: i+1 for i, word in enumerate(words)} # 0 reserved for padding


with open("data/processed/vocab.json", "w") as f:
    json.dump(vocab, f)


def text_to_indices(text):
    return [vocab[word] for word in text.split() if word in vocab]


df['indices'] = df['text'].apply(text_to_indices)


MAX_LEN = 5
def pad_sequence(seq):
    if len(seq) < MAX_LEN:
        return seq + [0] * (MAX_LEN - len(seq))
    else:
        return seq[:MAX_LEN]


X = list(map(pad_sequence, df['indices'].tolist()))
y = df['label'].tolist()


torch.save({'X': torch.tensor(X), 'y': torch.tensor(y).float()}, "data/processed/train_data.pt")

print("Data processed and saved to data/processed/train_data.pt")