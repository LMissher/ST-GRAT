import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import math


class DCN(nn.Module):
    def __init__(self, N, h, k):
        super(DCN, self).__init__()
        self.h = h
        self.k = k
        self.W = [nn.Parameter(torch.empty([N,N], dtype=torch.float32, device=device), requires_grad=True) for i in range(k*h)]
        for i in range(k*h):
            torch.nn.init.xavier_normal_(self.W[i])

    def forward(self, a):
        p = []
        for i in range(self.h):
            pi = self.W[i*self.k] * a[0]
            for j in range(1,self.k):
                pi += self.W[i*self.k + j] * a[j]
            p.append(pi)
        p = torch.stack(p, 0).unsqueeze(1).unsqueeze(1)
        return p

class Encoder_Decoder_Attention(nn.Module):
    def __init__(self, outfea, d):
        super(Encoder_Decoder_Attention, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.GELU(),
            nn.Linear(outfea, outfea),
        )

        self.ln = nn.LayerNorm(outfea)

        self.d = d

    def forward(self, x, xp):
        query = self.qff(x)
        key = self.kff(xp)
        value = self.vff(xp)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)

        a = torch.matmul(query, key)
        a /= (self.d ** 0.5)
        a = torch.softmax(a, -1)

        value = torch.matmul(a, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)
        value = self.ff(value) + x

        return self.ln(value)

class Temporal_Attention(nn.Module):
    def __init__(self, outfea, k, d):
        super(Temporal_Attention, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.GELU(),
            nn.Linear(outfea, outfea),
        )

        self.ln = nn.LayerNorm(outfea)

        self.d = d
        self.k = k

    def forward(self, x, Mask=False):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)

        a = torch.matmul(query, key)
        a /= (self.d ** 0.5)

        if Mask == True:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(device) # [T,T]
            mask = torch.tril(mask) # [T,T]但是对角线以上的值变成0了
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.k * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(a).to(device) # [k*B,N,T,T]里面元素全是负无穷大
            a = torch.where(mask, a, zero_vec)

        a = torch.softmax(a, -1)

        value = torch.matmul(a, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)
        value = self.ff(value) + x

        return self.ln(value)

class Spatial_Attention(nn.Module):
    def __init__(self, outfea, N, k, d, K):
        super(Spatial_Attention, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.ksff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)
        self.vsff = nn.Linear(outfea, outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.GELU(),
            nn.Linear(outfea, outfea),
        )

        self.ln = nn.LayerNorm(outfea)
        self.ln1 = nn.LayerNorm(outfea)

        self.d = d
        self.k = k

        self.dcni = DCN(N, k//2, K)
        self.dcno = DCN(N, k//2, K)

    def forward(self, x, A, AT):
        query = self.qff(x)
        key = self.kff(x)
        keys = self.ksff(x)
        value = self.vff(x)
        values = self.vsff(x)

        query = torch.stack(torch.split(query, self.d, -1), 0)
        key = torch.stack(torch.split(key, self.d, -1), 0)
        keys = torch.stack(torch.split(keys, self.d, -1), 0)
        value = torch.stack(torch.split(value, self.d, -1), 0)
        values = torch.stack(torch.split(values, self.d, -1), 0)

        # inflow
        e = torch.matmul(query[:self.k//2], key[:self.k//2].transpose(-1,-2)) / (self.d ** 0.5) + self.dcni(A)
        # e = torch.matmul(query[:self.k//2], key[:self.k//2].transpose(-1,-2)) / (self.d ** 0.5)
        es = torch.sigmoid((query[:self.k//2] * keys[:self.k//2]).sum(-1, keepdim=True) / (self.d ** 0.5))
        e = e - torch.max(e, -1, keepdim=True)[0]
        a = torch.exp(e) / (es + torch.exp(e).sum(-1, keepdim=True))
        # a = torch.softmax(e, -1)

        valuei = torch.matmul(a, value[:self.k//2]) + values[:self.k//2] - a.sum(-1, keepdim=True)*values[:self.k//2]

        # outflow
        eo = torch.matmul(query[self.k//2:], key[self.k//2:].transpose(-1,-2)) / (self.d ** 0.5) + self.dcno(AT)
        # eo = torch.matmul(query[self.k//2:], key[self.k//2:].transpose(-1,-2)) / (self.d ** 0.5)
        eso = torch.sigmoid((query[self.k//2:] * keys[self.k//2:]).sum(-1, keepdim=True) / (self.d ** 0.5))
        eo = eo - torch.max(eo, -1, keepdim=True)[0]
        ao = torch.exp(eo) / (eso + torch.exp(eo).sum(-1, keepdim=True))
        # ao = torch.softmax(eo, -1)

        valueo = torch.matmul(ao, value[self.k//2:]) + values[self.k//2:] - ao.sum(-1, keepdim=True)*values[self.k//2:]
        
        # cat
        value = torch.cat([valuei, valueo], 0)
        value = torch.cat(torch.split(value, 1, 0), -1).squeeze(0)
        value = self.ff(value) + x
        return self.ln(value)

class FeedForward(nn.Module):
    def __init__(self, outfea):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.GELU(),
            nn.Linear(outfea, outfea)
        )
        self.ln = nn.LayerNorm(outfea)

    def forward(self, x, xl):
        x = self.ff(x) + xl
        return self.ln(x)

class Emb(nn.Module):
    def __init__(self, outfea, max_len=12):
        super(Emb, self).__init__()
        self.ff = nn.Linear(outfea+64, outfea)

        pe = torch.zeros(max_len, outfea)
        for pos in range(max_len):
            for i in range(0, outfea, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/outfea)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / outfea)))
        pe = pe.unsqueeze(0).unsqueeze(2) #[1,T,1,F]
        self.register_buffer('pe', pe)

    def forward(self, x, se):
        # se = torch.from_numpy(se).to(device)
        se = se.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],x.shape[1],1,1)
        x = torch.cat([x, se], -1)
        x = self.ff(x)

        return x + Variable(self.pe[:,:x.shape[1],:,:], requires_grad=False)


class Decoder(nn.Module):
    def __init__(self, outfea, L, N, k, d, K):
        super(Decoder, self).__init__()
        self.spatial_attention = nn.ModuleList([Spatial_Attention(outfea, N,k,d,K) for i in range(L)])
        self.masked_temporal_attention = nn.ModuleList([Temporal_Attention(outfea, k, d) for i in range(L)])
        self.encoder_decoder_attention = nn.ModuleList([Encoder_Decoder_Attention(outfea, d) for i in range(L)])
        self.ff = nn.ModuleList([FeedForward(outfea) for i in range(L)])

        self.L = L

    def forward(self, x, xp, A, AT):
        for i in range(self.L):
            xl = x
            x = self.spatial_attention[i](x, A, AT)
            x = self.masked_temporal_attention[i](x, True)
            x = self.encoder_decoder_attention[i](x, xp)
            x = self.ff[i](x, xl)

        return x

class Encoder(nn.Module):
    def __init__(self, outfea, L, N, k, d, K):
        super(Encoder, self).__init__()
        self.spatial_attention = nn.ModuleList([Spatial_Attention(outfea, N,k,d,K) for i in range(L)])
        self.temporal_attention = nn.ModuleList([Temporal_Attention(outfea, k, d) for i in range(L)])
        self.ff = nn.ModuleList([FeedForward(outfea) for i in range(L)])

        self.L = L

    def forward(self, x, A, AT):
        for i in range(self.L):
            xl = x
            x = self.spatial_attention[i](x, A, AT)
            x = self.temporal_attention[i](x)
            x = self.ff[i](x, xl)
        return x

class STGRAT(nn.Module):
    def __init__(self, SE, infea, outfea, L, N, k, d, K, A, AT, dev):
        super(STGRAT, self).__init__()
        global device
        device = dev

        self.encoder = Encoder(outfea, L, N, k, d, K)
        self.decoder = Decoder(outfea, L, N, k, d, K)

        self.encode_start_emb = nn.Linear(infea, outfea)
        self.decode_start_emb = nn.Linear(infea, outfea)
        self.end_emb = nn.Linear(outfea, infea)

        self.src_emb = Emb(outfea)
        self.tgt_emb = Emb(outfea)

        self.se = SE
        self.A = A
        self.AT = AT

    def forward(self, xp, x):
        return self.decode(x, self.encode(xp))

    def encode(self, x):
        x = x.unsqueeze(-1)
        x = self.encode_start_emb(x)
        x = self.src_emb(x, self.se)
        x = self.encoder(x, self.A, self.AT)

        return x

    def decode(self, x, xp):
        x = x.unsqueeze(-1)
        x = self.decode_start_emb(x)
        x = self.tgt_emb(x, self.se)
        x = self.decoder(x, xp, self.A, self.AT)
        x = self.end_emb(x)

        return x.squeeze(-1)
