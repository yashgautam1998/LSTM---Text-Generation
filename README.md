🧠 LSTM Text Generator (Word-Level)

A word-level LSTM-based text generator trained on the Complete Works of Shakespeare from Project Gutenberg.
This script demonstrates natural language generation using a recurrent neural network in TensorFlow/Keras.

📘 Overview

This project builds a deep learning model that learns Shakespeare’s writing style and generates similar text word by word.

The pipeline includes:

Dataset download (Shakespeare’s works via Project Gutenberg)

Text preprocessing (cleaning, lowercasing, removing punctuation)

Tokenization and sequence creation (word-level)

Model training (Embedding → LSTM → Dense Softmax)

Text generation using a seed prompt

🧩 Features

Automatic dataset download and preprocessing

Word-level LSTM training pipeline

Model checkpointing and early stopping

Adjustable sequence length, embedding size, and temperature for sampling

Command-line interface for training and generation

🧠 Model Architecture
Embedding (vocab_size → 128)
        ↓
LSTM (256 units)
        ↓
Dense (softmax output over vocabulary)

🧰 Requirements

Install dependencies before running the project:

pip install tensorflow numpy requests tqdm

📂 Project Structure
lstm_text_generator/
│
├── lstm_text_generator.py   # Main script
├── data/
│   ├── shakespeare.txt      # Raw dataset (auto-downloaded)
│   ├── tokenizer.json       # Saved tokenizer
│   └── lstm_text_gen.h5     # Trained model
└── README.md                # Documentation

⚙️ Usage
1. Train the model
python lstm_text_generator.py --train


Downloads and preprocesses the Shakespeare dataset

Tokenizes text into word sequences

Trains the LSTM model with early stopping and checkpointing

Saves the model and tokenizer to the data/ folder

2. Generate text

Once training is complete, generate text using a seed phrase:

python lstm_text_generator.py --generate --seed "to be or not to be" --length 50


Optional arguments:

--seed → Starting text prompt

--length → Number of words to generate

--temperature → Controls creativity (default = 1.0).

Lower = safer / more predictable

Higher = more random / creative

🧪 Example Output
> python lstm_text_generator.py --generate --seed "love is" --length 20

love is not a man of war nor a friend but a poor heart that cannot speak for fear

🗃️ Saved Files
File	Description
data/shakespeare.txt	Raw dataset from Project Gutenberg
data/tokenizer.json	Tokenizer vocabulary used for encoding words
data/lstm_text_gen.h5	Trained model weights
🚀 Tips for Better Results

Train longer (e.g., 100+ epochs) with a GPU for better fluency

Increase the dataset size (use more Shakespeare works or combine other authors)

Experiment with temperature and sequence length for varied outputs

📄 License

This project uses public-domain text from Project Gutenberg
.
All generated outputs are free to use.
