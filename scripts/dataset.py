from torch.utils.data import Dataset, DataLoader
import torch
from tokenizers import Tokenizer #type:ignore
from typing import Literal
import pandas as pd #type:ignore

from train_tokenizer import load_tokenizer


def causal_mask(size):
    mask = torch.tril(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask  


def get_dataloder(config):
    data = pd.read_csv(config.data_path)
    data = data[(data['bn_token_len'] < 50) & (data['en_token_len'] < 50)]
    val_data = data.sample(frac=config.test_size)
    train_data = data.drop(val_data.index)
    val_data.reset_index(inplace=True)
    train_data.reset_index(inplace=True)

    src_tokenizer = load_tokenizer(config.src_lang)
    tgt_tokenizer = load_tokenizer(config.tgt_lang)
    train_dataset = LanguageTranslation(train_data, src_tokenizer, tgt_tokenizer, config.src_lang, config.tgt_lang, config.seq_len)
    val_dataset = LanguageTranslation(val_data, src_tokenizer, tgt_tokenizer, config.src_lang, config.tgt_lang, config.seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    return train_dataloader, val_dataloader


class LanguageTranslation(Dataset):
    def __init__(self, ds:pd.DataFrame, src_tokenizer:Tokenizer, tgt_tokenizer:Tokenizer, src_lang:str, tgt_lang:str, seq_len:int):
        
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor(self.src_tokenizer.token_to_id("[SOS]"), dtype=torch.int64).unsqueeze(0)
        self.eos_token = torch.tensor(self.src_tokenizer.token_to_id("[EOS]"), dtype=torch.int64).unsqueeze(0)
        self.pad_token = self.src_tokenizer.token_to_id("[PAD]")
        
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        print(f"Index : {index}")
        src_text = self.ds.loc[index, self.src_lang]
        tgt_text = self.ds.loc[index, self.tgt_lang]
        src_encoding = self.src_tokenizer.encode(src_text)
        tgt_encoding = self.tgt_tokenizer.encode(tgt_text)
        
        src_pad_len = self.seq_len - len(src_encoding.ids) - 2
        tgt_pad_len = self.seq_len - len(tgt_encoding.ids) - 1
        
        if src_pad_len < 0 or tgt_pad_len < 0:
            raise ValueError("Seq Length cannot be Less Token Length")
        
        encoder_input = torch.cat([
            self.sos_token.unsqueeze(0), 
            torch.tensor(src_encoding.ids, dtype=torch.int64), 
            self.eos_token.unsqueeze(0), 
            torch.tensor([self.pad_token]*src_pad_len, dtype=torch.int64)
        ])
        
        decoder_input = torch.cat([
            self.sos_token.unsqueeze(0),
            torch.tensor(tgt_encoding.ids, dtype=torch.int64),
            torch.tensor([self.pad_token]*tgt_pad_len, dtype=torch.int64)
        ])
        
        label = torch.cat([
            torch.tensor(tgt_encoding.ids, dtype=torch.int64),
            self.eos_token.unsqueeze(0),
            torch.tensor([self.pad_token]*tgt_pad_len, dtype=torch.int64)
        ])

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            'encoder_input':encoder_input,
            'decoder_input':decoder_input,
            'encoder_mask':encoder_mask,
            'decoder_mask':decoder_mask,
            'label':label,
            'src_text':src_text,
            'tgt_text':tgt_text
        }

