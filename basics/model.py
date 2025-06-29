from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int


from basics.nn_utils import softmax
from dataclasses import dataclass
logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self,d_in:int,d_out:int):
        super().__init__()
        std=math.sqrt(2/(d_in+d_out))
        self.weight:Float[Tensor,"d_out d_in"]=nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out,d_in),std=std,a=-3*std,b=3*std),
            requires_grad=True
        )
    def forward(self,x:Float[Tensor,"... d_in"]):
        return einsum(x,self.weight,"... d_in,d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"d_out={self.weight.shape[0]},d_in={self.weight.shape[1]}"

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        d_model:int
    ):
        super().__init__()
        std=1.0
        self.weight=nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size,d_model),std=std,a=-3*std,b=3*std),
            requires_grad=True
        )
    def forward(self,token_ids:Int[Tensor," ..."])->Float[Tensor," ... d_model"]:
        return self.weight[token_ids,:]
    def extra_repr(self) -> str:
        return f"vocab_size={self.weight.shape[0]},d_model={self.weight.shape[1]}"
    
class RMSnorm(nn.Module):
    def __init__(
        self,
        hidden_size:int,
        eps:float=1e-5,
        device=None,
    ):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size,device=device))
        self.eps=eps
    def forward(self,x:Float[Tensor, "batch_size,*"]):
        in_dtype=x.dtype

        x=x.to(torch.float32)
        rms=torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        x=x*rms
        return (self.weight * x).to(in_dtype)
class RotaryEmbedding(nn.Module):
    def __init__(self,
    context_length:int,
    dim:int,
    theta:float=10000.0
    ):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length,dim,theta),persistent=False
        )
    @staticmethod
    def _init_cache(context_length:int,dim:int,theta:float)->Float[Tensor," 2 context_length half_dim"]:
        assert dim%2==0

        d=torch.arange(0,dim,2)/dim
        freqs=theta ** -d
        t=torch.arange(context_length)
        freqs=einsum(t,freqs,"t,f ->t f")
        cos,sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos,sin))
    
    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result

def scale_dot_attention(
        Q:Float[Tensor," ... queries d_k"],
        K:Float[Tensor," ... keys d_k"],
        V:Float[Tensor," ... keys d_v"],
        mask:Bool[Tensor,"... queries keys"]|None=None,
)->Float[Tensor,"... queries d_v"]:
    d_k=K.shape[-1]
    attention_scores=einsum(Q,K,"... query d_k,... key d_k-> ... query key")/math.sqrt(d_k)
    if mask is not None:
        attention_scores=torch.where(mask,attention_scores,float("-inf"))
    attention_weights = softmax(attention_scores,dim=-1)
    return einsum(attention_weights,V,"... query key,... key d_v-> ... query d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,positional_encoder:RotaryEmbedding):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads

        self.d_k=d_model//num_heads
        self.d_v=self.d_k

        self.q_proj=Linear(self.d_model,self.num_heads*self.d_k)
        self.k_proj=Linear(self.d_model,self.num_heads*self.d_k)
        self.v_proj=Linear(self.d_model,self.num_heads*self.d_v)
        self.out_proj=Linear(self.num_heads*self.d_v,self.d_model)
        self.positional_encoder = positional_encoder

    def forward(self,x:Float[Tensor," ... seq d_k"],token_positions:Int[Tensor," ... seq"]|None=None)->Float[Tensor,"... seq d_v"]:
        
        *b, sequence_length, d_model = x.size()
        assert d_model==self.d_model
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)
        Q,K,V=(
            rearrange(X,"... seq (heads d)-> ... heads seq d",heads=self.num_heads)
            for X in (Q,K,V)
        )
        if token_positions is None:
            token_positions=einx.rearrange("seq->b... seq",torch.arange(sequence_length,device=x.device),b=[1]*len(b))
        
        token_positions=rearrange(token_positions,"...  seq->... 1 seq")

        Q=self.positional_encoder(Q,token_positions)
        K=self.positional_encoder(K,token_positions)

        seq=torch.arange(sequence_length,device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask=qi>=kj
        attn_output = scale_dot_attention(Q=Q,K=K,V=V,mask=causal_mask)
        attn_output=rearrange(attn_output,'batch heads seq d_v -> batch seq (heads d_v)').contiguous()

        output=self.out_proj(attn_output)
        return output
    
def silu(x:Tensor):
    return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff:int):
        super().__init__()
        self.w1=Linear(d_model,d_ff)
        self.w2=Linear(d_ff,d_model)
        self.w3=Linear(d_model,d_ff)
    def forward(self,x):
        return self.w2(self.w1(x)*self.w3(x))            
        
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,positional_encoder):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.positional_encoder=positional_encoder
        d_q,d_k,d_v=d_model,d_model,d_model

        self.ln=RMSnorm(self.d_model)
        self.attn_norm=RMSnorm(self.d_model)
        self.attn=CausalMultiHeadSelfAttention(d_model=d_model,num_heads=num_heads,positional_encoder=positional_encoder)
        self.FFN=SwiGLU(d_model=d_model,d_ff=d_ff)
    def forward(self,x):
        x=x +self.attn(self.attn_norm(x))
        
        x=x + self.FFN(self.ln(x))
        return x
   
class BasicTransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            context_length:int,
            d_model:int,
            num_layers:int,
            num_heads:int,
            d_ff:int,
            rope_theta:float=10000.0,
            **kwargs
        ):
        self.config={
            k:v for k,v in locals().items() if k!="self" and not (k.startswith("__")) and k.endswith("__")
        }
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.token_embeddings=Embedding(vocab_size,d_model)
        d_head=d_model//num_heads
        self.positional_encoder=RotaryEmbedding(
            context_length=context_length,
            dim=d_head,
            theta=rope_theta
        )
        self.layers=nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder
                )
            ]
        )
        self.ln_final=RMSnorm(d_model)
        self.lm_head=Linear(d_model,vocab_size)

        logger.info(f"number of non-embeding parameters: {self.get_num_parameters()/1e6:.2f}")

    def get_num_parameters(self,non_embdedding=True):
        n_params=sum(p.numel() for p in self.parameters())
        if non_embdedding:
            n_params-=self.lm_head.weight.numel()
        return n_params
    
    def forward(self,x):
        x=self.token_embeddings(x)
        for layer in self.layers:
            x=layer(x)
        x=self.ln_final(x)
        x=self.lm_head(x)
        return x
    
    @torch.no_grad()
    def generate(
        self,
        x:torch.Tensor,
        max_new_tokens:int,
        temperature:float=1.0,
        top_k:int|None=None,
        eos_token_id:int|None=None,
    ):
        if x.dim()==1:
            x=x.unsqueeze(0)
        original_sequence_length=x.size(-1)
        for _ in range(max_new_tokens):
            x=x[:,-self.context_length:] if x.size(-1)>self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:,-1]
            temperature_scaled_next_token_logits=next_token_logits/temperature
            if top_k:
                topk_values=torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k,temperature_scaled_next_token_logits.size(-1))
                )
                threshold=topk_values[:,-1]
                topk_mask=temperature_scaled_next_token_logits<threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities=softmax(temperature_scaled_next_token_logits,dim=-1)
            next_token_id=torch.multinomial(next_token_probabilities,1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")

        checkpoint = torch.load(weights_path)
        state_dict=checkpoint['model_state_dict']
        
        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        return model