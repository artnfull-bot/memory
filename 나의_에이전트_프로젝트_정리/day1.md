# Transformer 구현 코드 (PyTorch)

"Attention Is All You Need" (Vaswani et al., 2017) 논문을 기반으로 구현한 기본적인 트랜스포머(Transformer) 모델입니다.

```python
import torch
import torch.nn as nn
import math

# 1. 스케일드 닷 프로덕트 어텐션 (Scaled Dot-Product Attention)
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V 차원: [batch_size, num_heads, seq_len, d_k/d_v]
        
        # (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # 마스크가 있는 경우, 0인 부분을 매우 작은 값(-1e9)으로 채워 softmax 결과가 0이 되게 함
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax를 통해 어텐션 가중치 계산
        attn = torch.softmax(scores, dim=-1)
        
        # 가중치와 V 곱하기
        output = torch.matmul(attn, V)
        return output, attn


# 2. 멀티 헤드 어텐션 (Multi-Head Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V를 위한 선형 변환
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 최종 출력을 위한 선형 변환
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 선형 변환 및 헤드 수에 맞게 차원 변환: [batch_size, num_heads, seq_len, d_k]
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            # mask 차원을 헤드 수에 맞게 확장
            mask = mask.unsqueeze(1)
            
        # 어텐션 계산
        output, attn = self.attention(q, k, v, mask)
        
        # 분리된 헤드를 다시 하나로 합치기: [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 최종 선형 변환
        output = self.W_o(output)
        return output


# 3. 포지션 와이즈 피드 포워드 네트워크 (Position-wise Feed-Forward Network)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# 4. 포지셔널 인코딩 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # 모델 저장 시 같이 저장되지만 학습되는 파라미터는 아님

    def forward(self, x):
        # x 차원: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


# 5. 인코더 레이어 (Encoder Layer)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. Multi-Head Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# 6. 디코더 레이어 (Decoder Layer)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked Multi-Head Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Cross Multi-Head Attention + Add & Norm
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 3. Feed Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# 7. 전체 트랜스포머 모델 (Transformer Model)
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 임베딩 레이어
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 인코더와 디코더 레이어들을 담을 리스트
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src, pad_idx):
        # src_mask: [batch_size, 1, seq_len]
        src_mask = (src != pad_idx).unsqueeze(1)
        return src_mask

    def make_tgt_mask(self, tgt, pad_idx):
        # tgt_pad_mask: [batch_size, 1, seq_len]
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1)
        
        seq_len = tgt.size(1)
        # tgt_sub_mask: 하 삼각 행렬 생성하여 미래 시점의 단어를 보지 못하게 함
        tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        
        # pad 마스크와 sub 마스크를 합침
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        # 마스크 생성
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)
        
        # 인코더
        enc_output = self.dropout(self.pos_encoding(self.src_emb(src)))
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            
        # 디코더
        dec_output = self.dropout(self.pos_encoding(self.tgt_emb(tgt)))
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # 최종 출력값 (vocab size에 대한 로짓값)
        output = self.fc_out(dec_output)
        
        return output

# 모델 테스트 예시
if __name__ == '__main__':
    # 설정값
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 32
    seq_len = 50
    pad_idx = 0
    
    # 임의의 데이터 생성
    src = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))
    
    # 모델 초기화
    model = Transformer(src_vocab_size, tgt_vocab_size)
    
    # 순전파 진행
    out = model(src, tgt, src_pad_idx=pad_idx, tgt_pad_idx=pad_idx)
    
    print(f"출력 텐서 형태: {out.shape}") 
    # 예상 형태: [batch_size, seq_len, tgt_vocab_size] -> [32, 50, 10000]
```
