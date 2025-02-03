from typing import Any, Callable, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
import os



class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""
    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift

class GatedUnits(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sigmoid()
        
    def forward(self, t, text):
        gated_t = self.gate(t)
        gated_text = self.gate(text)
        combined = gated_t * t + gated_text * text
        return combined

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift



class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation


        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.gated_units = GatedUnits(d_model)
        
        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        tgt,
        memory,
        t,
        text,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            combined_condition = self.gated_units(t, text)
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(combined_condition))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t, text):
        for layer in self.stack:
            x = layer(x, cond, t, text)
        return x



class SeqModel(nn.Module):
    def __init__(self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 512,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:
        super().__init__()
     
        self.cond_hand = nn.Linear(cond_feature_dim , 32)
     
        self.network = nn.ModuleDict()
        self.network['body_net'] = DanceDecoder(
            nfeats=4+3+22*6,
            seq_len=seq_len,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=512+193,
            activation=activation
        )
        self.network['hand_net'] = DanceDecoder_hand(
            nfeats=30*6,
            seq_len=seq_len,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=32+139,    
            activation=activation
        )


    def forward(self, x: Tensor, cond_embed: Tensor, times: Tensor, text: Tensor,cond_drop_prob: float = 0.0):
        x_body_start = x[:,:,:4+135]
        x_hand_start = x[:,:,4+135:]
        body_output = self.network['body_net'](x_body_start, cond_embed, times, text, cond_drop_prob)
        cond_music = self.cond_hand(cond_embed)
        cond_embed = torch.cat([body_output, cond_music], dim = -1)
        hand_output = self.network['hand_net'](x_hand_start, cond_embed, times, text, cond_drop_prob)
        output = torch.cat([body_output, hand_output], dim=-1)   
        return output

    def guided_forward(self, x, cond_embed, times, text, guidance_weight):
        unc = self.forward(x, cond_embed, times, text, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, text, cond_drop_prob=0)
        return unc + (conditioned - unc) * guidance_weight


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 120, 
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 512+193,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats
        self.rotary = None
        self.cond_mlp_basic = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.cond_basic = nn.Linear(193, latent_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.abs_pos_encoding = nn.Identity()
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )
        
        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),   # pos embedding
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        # conditional projection
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)


    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor,  text: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device
        x = self.input_projection(x)           
        x = self.abs_pos_encoding(x)
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_basic = self.cond_basic(cond_embed[:,:,512:])
        cond_embed =torch.add(cond_embed[:,:,:512],cond_basic)
        cond_tokens = self.cond_mlp_basic(cond_embed)
        
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        
        text_tokens = self.text_mlp(text)


        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        t_hidden = self.time_mlp(times)

        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)    #16 x 2 x 512

        null_cond_hidden = self.null_cond_hidden.to(t.dtype)    
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        text_tokens = text_tokens.unsqueeze(1)
        c = torch.cat((cond_tokens,text_tokens,t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        output = self.seqTransDecoder(x, cond_tokens, t, text)
        output = self.final_layer(output)
        return output


class DanceDecoder_hand(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 35,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats
        # positional embeddings
        self.rotary = None
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_feature_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.text_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
        self.abs_pos_encoding = nn.Identity()
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        
        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)


    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor,  text: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device
        # project to latent space
        x = self.input_projection(x)    
        x = self.abs_pos_encoding(x)        #batch x seq x 512
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_mlp(cond_embed)
        text_tokens = self.text_mlp(text)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)



        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        t_hidden = self.time_mlp(times)

        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)   

        null_cond_hidden = self.null_cond_hidden.to(t.dtype)    
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        text_tokens = text_tokens.unsqueeze(1)
        c = torch.cat((cond_tokens,text_tokens,t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        output = self.seqTransDecoder(x, cond_tokens, t, text)
        output = self.final_layer(output)
        return output
