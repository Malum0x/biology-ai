### this is a file to create the architecture / empty structure / skeleton - for gpt model - taken from the karpathy nanogpt - just for studying and understanding

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ Normalizes values to have mean=0, variance=1;
        stabilizes training, helps gradients flow;
        Applied before attention, and need feed-forward layers
        has learnable parameters: scale (gamma), and shift (batch) """
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim)) # scale parameter, initialized to ones (so no scaling at start)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None # shift parameter, initialized to zeros (so no shift at start)

    def forward(self, input):
        # 1e-5 is epsilon, prevents division by zero
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CausalSelfAttention(nn.Module):
    """ It's the core attention layer, where the model decides which previous tokens to pay attention to when predicting the next one;
        Causal means: a token can look at tokens before it, not after it. this is what makes it autoregressive, predicting left to the right, 
        without cheating """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0 # embedding dimension must divide evenly across heads, so each head gets equal slice (modulo %)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # the linear layer that creates Q, K, V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) # Output Projection
        self.attn_dropout = nn.Dropout(config.dropout) #Dropout randomly set some values to zero during training, prevents overfitting 
        self.resid_dropout = nn.Dropout(config.dropout) #Prevents relying on specific sublayer outputs
        self.n_head = config.n_head # set n_head
        self.n_embd = config.n_embd # set n_embd
        self.dropout = config.dropout # set dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') # checks if fast flash attn is available (pytorch 2.0+)
        if not self.flash:
            print("Warning: using slow attention, flash attention requires python >= 2.0")
            self.register_buffer("bias", torch.trill(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
            # causal mask: lower triangle of 1s,
            # prevents tokens from attending to future positions
            # register_buffer = saved with model but not trained
            # only needed when flash attention unavailable (flash handles masking internally)
            # this is what makes it causal 
    
    def forward(self, x):
        # input -> make Q, K, V -> Split heads -> Attention math -> Combine heads -> Outputs
        B, T, C = x.size() # batch_size, sequence_length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # cause self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using flash attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs size by size

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):
    """ Multi Layer PerceptRon - feed-forward network that processes each position independietly """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # linear layer, expands dimensions
        self.gelu = nn.GELU() #Activation Function 
        self.c_proj = nn.Linear(4 * config,n_embd, config.n_embd, bias=config.bias) # linear layer that projects output back
        self.dropout = nn.Dropout(config.dropout) # dropout is known 

    def forward(self, x): 
        x = self.c_fc(x) # (B, T, C) → (B, T, 4*C)  expand
        x = self.gelu(x) # activation (no shape change)
        x = self.c_proj(x) # (B, T, 4*C) → (B, T, C)  shrink back
        x = self.dropout(x) # regularization
        return x

class Block(nn.Module):
    """ One transformer block - combines attention + MLP 
        NanoGPT uses pre-norm (is more stable for training) """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # first layer norm 
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias) # second layer norm
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # normalize -> attention -> add back
        x = x + self.mlp(self.ln_2(x)) # normalize -> mlp -> add back
        return x 

@dataclass
 # continue tomorrow