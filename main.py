"""
LSTM Text Generator (word-level) - lstm_text_generator.py

This script downloads a public-domain text (Complete Works of Shakespeare by Project Gutenberg),
preprocesses it (lowercase, remove punctuation), tokenizes into words, creates input-output
sequences, builds an LSTM model (Embedding -> LSTM -> Dense softmax), trains with checkpoints
and early stopping, and contains text-generation utilities.

Usage:
    python lstm_text_generator.py --train    # train the model
    python lstm_text_generator.py --generate --seed "to be or not to be" --length 50

Notes:
 - This is a demonstration. For full training, use a machine with a GPU and increase epochs.
 - The script saves tokenizer and model to disk.

Requirements:
    tensorflow (>=2.10), numpy, requests, tqdm

"""

import os
import argparse
import re
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_TXT = DATA_DIR / "shakespeare.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
MODEL_PATH = DATA_DIR / "lstm_text_gen.h5"

SEQUENCE_LENGTH = 25  # number of input words
EMBEDDING_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 128
EPOCHS = 50

# -----------------------------
# Helpers: download dataset
# -----------------------------
PGUT_URL = "https://www.gutenberg.org/files/100/100-0.txt"  # Project Gutenberg "Complete Works" UTF-8


def download_dataset(force=False):
    if RAW_TXT.exists() and not force:
        print(f"Dataset already exists at {RAW_TXT}")
        return RAW_TXT
    import requests
    print("Downloading dataset from Project Gutenberg...")
    r = requests.get(PGUT_URL, timeout=30)
    r.raise_for_status()
    text = r.text
    RAW_TXT.write_text(text, encoding="utf-8")
    print(f"Saved raw text to {RAW_TXT}")
    return RAW_TXT

# -----------------------------
# Preprocessing
# -----------------------------

def load_and_preprocess(path=RAW_TXT):
    text = path.read_text(encoding="utf-8")
    # Optionally remove Gutenberg header/footer
    # Keep only between *** START OF THIS PROJECT GUTENBERG EBOOK *** and END
    start_match = re.search(r"\*\*\* START OF.*?\*\*\*", text, flags=re.IGNORECASE|re.DOTALL)
    end_match = re.search(r"\*\*\* END OF.*?\*\*\*", text, flags=re.IGNORECASE|re.DOTALL)
    if start_match and end_match:
        text = text[start_match.end():end_match.start()]

    # Lowercase
    text = text.lower()
    # Remove all punctuation except apostrophes inside words
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Tokenization & sequences
# -----------------------------

def make_tokenizer_and_sequences(text, seq_len=SEQUENCE_LENGTH):
    # Simple word tokenizer using Keras TextVectorization or Tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(filters='')  # we've already cleaned punctuation
    tokenizer.fit_on_texts([text])
    word_index = tokenizer.word_index
    total_words = len(word_index) + 1
    print(f"Vocabulary size: {total_words}")

    # Convert to sequence of word ids
    tokens = tokenizer.texts_to_sequences([text])[0]

    input_sequences = []
    targets = []
    for i in range(seq_len, len(tokens)):
        seq = tokens[i-seq_len:i]
        target = tokens[i]
        input_sequences.append(seq)
        targets.append(target)

    X = np.array(input_sequences)
    y = np.array(targets)
    # Save tokenizer
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json()))

    return tokenizer, X, y, total_words

# -----------------------------
# Model
# -----------------------------

def build_model(vocab_size, seq_len=SEQUENCE_LENGTH, embed_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_len),
        LSTM(lstm_units, return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# -----------------------------
# Generation
# -----------------------------

def load_tokenizer():
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer JSON not found. Run training first.")
    tokenizer_json = TOKENIZER_PATH.read_text(encoding='utf-8')
    tokenizer = tokenizer_from_json(json.loads(tokenizer_json))
    return tokenizer


def generate_text(seed_text, next_words=50, temperature=1.0):
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    model = tf.keras.models.load_model(MODEL_PATH)

    result = seed_text.split()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([" ".join(result[-SEQUENCE_LENGTH:])])[0]
        # pad if needed
        if len(token_list) < SEQUENCE_LENGTH:
            token_list = [0] * (SEQUENCE_LENGTH - len(token_list)) + token_list
        token_arr = np.array([token_list])
        preds = model.predict(token_arr, verbose=0)[0]
        # apply temperature
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-9) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        next_index = np.argmax(probas)
        if next_index == 0:
            # unknown or padding, break
            break
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                result.append(word)
                break
    return " ".join(result)

# -----------------------------
# Training orchestration
# -----------------------------

def train(train_args):
    download_dataset()
    text = load_and_preprocess(RAW_TXT)
    tokenizer, X, y, vocab_size = make_tokenizer_and_sequences(text, seq_len=SEQUENCE_LENGTH)

    # Shuffle & split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    val_split = int(0.1 * len(X))
    X_train, X_val = X[val_split:], X[:val_split]
    y_train, y_val = y[val_split:], y[:val_split]

    model = build_model(vocab_size, seq_len=SEQUENCE_LENGTH)

    callbacks = [
        ModelCheckpoint(str(MODEL_PATH), monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--seed', type=str, default='to be or not to be')
    parser.add_argument('--length', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.generate:
        print("Generating text...")
        print(generate_text(args.seed, next_words=args.length, temperature=args.temperature))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

# -----------------------------
# End of file
# -----------------------------
