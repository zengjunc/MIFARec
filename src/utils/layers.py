# -*- coding: UTF-8 -*-
# @Author  : Anonymous
# @Email   : Anonymous


import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True, attention_d=-1):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        if attention_d<0:
            self.attention_d = self.d_model
        else:
            self.attention_d = attention_d

        self.d_k = self.attention_d // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, self.attention_d, bias=bias)
        self.k_linear = nn.Linear(d_model, self.attention_d, bias=bias)
        self.v_linear = nn.Linear(d_model, self.attention_d, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()

        # perform linear operation and split into h heads
        if not self.kq_same:
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
        output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(list(origin_shape)[:-1]+[self.attention_d]) # modified
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None):
        """
        This is called by Multi-head attention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        context = self.masked_attn_head(seq, seq, seq, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output


class TransformerLayer_Rt(nn.Module):
    def __init__(self, d_model, d_ff, d_t, n_heads, dropout=0, kq_same=False):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.linear_t1 = nn.Linear(d_model, d_ff)
        self.linear_t2 = nn.Linear(d_ff, d_t)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        # Self-attention layer
        context = self.masked_attn_head(seq, seq, seq, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)

        # Feedforward network (FFN) layer
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)

        # Frequency domain analysis layer
        context = self.layer_norm2(context)
        output_t = self.linear_t1(context).relu()
        output_t = self.linear_t2(output)
        output_t = torch.softmax(output_t, dim=-1)

        return output, output_t
