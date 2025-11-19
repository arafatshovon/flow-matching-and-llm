import os
import pandas as pd #type:ignore
from typing import Literal

from tokenizers import Tokenizer #type:ignore
from tokenizers.models import WordLevel #type:ignore
from tokenizers.trainers import WordLevelTrainer #type:ignore
from tokenizers.pre_tokenizers import Whitespace #type:ignore
from datasets import load_dataset #type:ignore


def download_dataset(save_dir:str):
    if not os.path.exists(save_dir):
        raise FileNotFoundError("Directory Does not Exist")
    
    print("Downloading Dataset...")
    data = load_dataset('Helsinki-NLP/opus-100', 'bn-en')
    data['train'].to_csv(f"{save_dir}/train_data.csv")
    print("Download Finished. Processing Started.....")
    
    df = pd.read_csv('./data/train_data.csv')
    df['translation_dict'] = df['translation'].apply(eval)

    df['bn'] = df['translation_dict'].apply(lambda x: x['bn'])
    df['en'] = df['translation_dict'].apply(lambda x: x['en'])

    df = df.drop('translation_dict', axis=1)
    file_path = f"{save_dir}/bangla_english_translation.csv"
    print("Processing Finished...")
    df.to_csv(file_path, index=False)
    return file_path


def train_and_save_tokenizer(file_path:str, output_path:str, vocab_size:int, lang:str = Literal['bn', 'en']):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File Path Does Not Exist")
    
    df = pd.read_csv(file_path)
    df = df.dropna()
    training_data = df.loc[:, lang].values
    
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=vocab_size,
                               min_frequency=2,
                               special_tokens=["[UNK]", "[PAD]", "[EOS]", "[SOS]"])
    
    print(f"Tokenizer Training Started for Language: {lang}")
    tokenizer.train_from_iterator(training_data, trainer)
    print("Tokenizer Training Finished...")
    
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(f"{output_path}/{lang}_tokenizer.json")


def load_tokenizer(lang:Literal['en', 'bn']):
    if lang not in ['en', 'bn']:
        raise ValueError("Language must be either 'en' or 'bn'")
    tokenizer = Tokenizer.from_file(f"./Tokenizer/{lang}_tokenizer.json")
    return tokenizer

if __name__ == "__main__":
    save_dir = "./data"
    file_path = download_dataset(save_dir)
    train_and_save_tokenizer(file_path, "./Tokenizer", vocab_size=1000, lang="bn")
    train_and_save_tokenizer(file_path, "./Tokenizer", vocab_size=1000, lang="en")
    
    print("Testing Trained Tokenizer:")
    
    en_tokenizer = load_tokenizer('en')
    sample_text = input("Enter a text:")
    encoding = en_tokenizer.encode(sample_text)
    print(f"Encoding IDS: {encoding.ids}")
    print(f"Encoding tokens: {encoding.tokens}")