import torch


from model import build_transformer
from dataset import get_dataloder

class config:
    data_path = './data/bangla_english_translation.csv'
    test_size = 0.2
    batch_size = 10
    epcohs = 10
    lr = 1e-4
    src_lang = 'en'
    tgt_lang = 'bn'
    src_vocab_size = 2000
    tgt_vocab_size = 2000
    seq_len = 15
    d_model = 512
    N = 6
    h = 8
    
    
    
model = build_transformer(config.src_vocab_size,
                        config.tgt_vocab_size,
                        config.seq_len,
                        config.seq_len,
                        config.d_model,
                        config.N,
                        config.h  
                        )

train_dataloader = get_dataloder(config)


