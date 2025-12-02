import torch
import torch.nn as nn
import math
from typing import Optional

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(10000.0, -1*torch.arange(0, d_model, 2).float/self.d_mdoel)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return x

class LayerNomalization(nn.Module):
    def __init__(self, epsilon:float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = torch.mean(dim=-1, keepdim=True)
        std = torch.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (self.std + self.epsilon) + self.bias

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forawrd(self, x):
        return self.linear2(self.dropout(self.linear1(x)))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h_dim:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h_dim = h_dim
        assert d_model % h_dim == 0, "embedding dimension is not divisible by number of heads"

        self.d_k = d_model // h_dim
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query:torch.tensor, key:torch.tensor, value:torch.tensor, mask:Optional[torch.tensor], dropout:Optional[nn.Linear] = None):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, 1e-9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores

    def forward(self, q:torch.tensor, k:torch.tensor, v:torch.tensor, mask:Optional[torch.tensor] = None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        B, T, _ = query.shape
        query = query.view(B, T, self.h_dim, self.d_k).transpose(1, 2)
        key = key.view(B, T, self.h_dim, self.d_k).transpose(1, 2)
        value = value.view(B, T, self.h_dim, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(B, T, self.h_dim*self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNomalization()
    
    def forward(self, x:torch.tensor, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))

class EncoderBlock(nn.Module):
    def __init__(self, attention_block:MultiHeadAttention, ffn:FeedForwardNetwork, dropout:float):
        super().__init__()

        self.attention_block = attention_block
        self.ffn = ffn
        self.res_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.res_conn[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.res_conn[1](x, self.ffn)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNomalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
        
class DecoderBlock(nn.Module):
    def __init__(self, self_attn_block:MultiHeadAttention, cross_attn_block:MultiHeadAttention, ffn:FeedForwardNetwork, dropout:float):
        super().__init__()
        self.self_attn_block = self_attn_block
        self.cross_attn_block = cross_attn_block
        self.ffn = ffn
        self.res_conn = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.res_conn[0](x, lambda x: self.self_attn_block(x, x, x, tgt_mask))
        x = self.res_conn[1](x, lambda x: self.cross_attn_block(x, encoder_output, encoder_output, src_mask))
        x = self.ffn(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNomalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.softmax(self.linear(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, enc:Encoder, dec:Decoder, src_emb:InputEmbedding, tgt_emb:InputEmbedding, src_pos:PositionalEmbedding, tgt_pos:PositionalEmbedding, proj:ProjectionLayer):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, x, src_mask):
        src = self.src_emb(x)
        src = self.src_pos(src)
        return self.enc(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer










