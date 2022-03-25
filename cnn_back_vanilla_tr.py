import torch as th
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.nn import ModuleList
import copy
import numpy as np
import make_beam_challenge
import os
import pickle
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4).cuda()
        self.w_2 = nn.Linear(d_model * 4, d_model).cuda()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout).cuda()

    def forward(self, x, sublayer):
        return x + self.dropout(F.relu(sublayer(self.norm(x))))


def self_attention(query, key, value, mask=None):
    key_transpose = torch.transpose(key, -2, -1)  # (bath, head_num, d_k, token_)
    matmul_result = torch.matmul(query, key_transpose)  # MatMul(Q,K)
    d_k = query.size()[-1]
    attention_score = matmul_result / math.sqrt(d_k)  # Scale

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e20)

    softmax_attention_score = F.softmax(attention_score, dim=-1)  # attention value
    result = torch.matmul(softmax_attention_score, value)

    return result, softmax_attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num=8, d_model=512, dropout=0.3):
        super(MultiHeadAttention, self).__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num

        self.w_q = nn.Linear(d_model, d_model).cuda()
        self.w_k = nn.Linear(d_model, d_model).cuda()
        self.w_v = nn.Linear(d_model, d_model).cuda()
        self.w_o = nn.Linear(d_model, d_model).cuda()

        self.self_attention = self_attention
        self.dropout = nn.Dropout(p=dropout).cuda()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.constant_(self.w_q.bias, 0)
        nn.init.constant_(self.w_k.bias, 0)
        nn.init.constant_(self.w_v.bias, 0)
        nn.init.constant_(self.w_o.bias, 0)

    def forward(self, query, key, value, mask=None, cache=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        eco = query.clone()
        batch_num = query.size(0)
        # print(query.shape)
        query = self.w_q(query).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)


        if cache is not None:
            key = torch.cat([cache["key"].type(key.dtype).cuda(), key], dim=2)
            value = torch.cat([cache["value"].type(value.dtype).cuda(), value], dim=2)

            cache["key"] = key.contiguous()
            cache["value"] = value.contiguous()
        # print(query.shape, key.shape)
        attention_result, attention_score = self.self_attention(query, key, value, mask)

        attention_result = attention_result.transpose(1, 2).contiguous().view(batch_num, -1, self.head_num * self.d_k)

        return self.w_o(attention_result)


class MultiHeadAttentionStack(nn.Module):
    def __init__(self, head_num=8, d_model=512, dropout=0.3):
        super(MultiHeadAttentionStack, self).__init__()
        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num

        self.w_q = nn.Linear(d_model, d_model).cuda()
        self.w_k = nn.Linear(d_model, d_model).cuda()
        self.w_v = nn.Linear(d_model, d_model).cuda()
        self.w_o = nn.Linear(d_model, d_model).cuda()

        self.self_attention = self_attention
        self.dropout = nn.Dropout(p=dropout).cuda()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.constant_(self.w_q.bias, 0)
        nn.init.constant_(self.w_k.bias, 0)
        nn.init.constant_(self.w_v.bias, 0)
        nn.init.constant_(self.w_o.bias, 0)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_num = query.size(0)

        k = query.shape[1]
        query = self.w_q(query).view(batch_num, k, -1, self.head_num, self.d_k).transpose(2, 3)
        key = self.w_k(key).view(batch_num, k, -1, self.head_num, self.d_k).transpose(2, 3)
        value = self.w_v(value).view(batch_num, k, -1, self.head_num, self.d_k).transpose(2, 3)

        attention_result, attention_score = self.self_attention(query, key, value, None)

        attention_result = attention_result.transpose(2, 3).contiguous().view(batch_num, -1, self.head_num * self.d_k)

        return self.w_o(attention_result)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


class Encoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, layers_num, max_length=360):
        super(Encoder, self).__init__()
        self.layers_num = layers_num
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(Encoderlayers(d_model, head_num, dropout)) for _ in range(self.layers_num)])
        # self.time_encoding = _gen_timing_signal(layers_num, d_model)
        self.positional_encoding = _gen_timing_signal(max_length, d_model)
        self.input_drop = nn.Dropout(p=dropout)
        self.layernorm = LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, inputs, mask, seg = None, pos_emb = None):
        inputs *= math.sqrt(self.d_model)
        # print(inputs.shape)
        inputs += self.positional_encoding[:, :inputs.shape[1], :].type_as(inputs.data)
        x = self.input_drop(inputs)
        for _ in range(self.layers_num):
            x = self.encoder_layers[_](x, mask)
        return self.layernorm(x)


class Encoderlayers(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Encoderlayers, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)

    def forward(self, input, mask):
        x = self.residual_1.forward(input, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2.forward(x, lambda x: self.feed_forward(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, layers_num, max_length=20):
        super(Decoder, self).__init__()
        self.layers_num = layers_num
        self.decoder_layers = nn.ModuleList(
             [copy.deepcopy(DecoderlayerTrOrigin(d_model, head_num, dropout)) for _ in range(self.layers_num)]
        )
        self.time_encoding = _gen_timing_signal(layers_num, d_model)
        self.positional_encoding = _gen_timing_signal(max_length, d_model)
        self.input_drop = nn.Dropout(p=dropout)
        self.d_model = d_model
    def forward(self, target, encoder_output1, target_mask, encoder_mask1, cache=None):
        target *= math.sqrt(self.d_model)
        target += self.positional_encoding[:, :target.shape[1], :].type_as(target.data)
        x = self.input_drop(target)
        for _ in range(self.layers_num):
            layer_name = "layer_%d" % _
            layer_cache = cache[layer_name] if cache is not None else None
            x = self.decoder_layers[_](x, encoder_output1, target_mask, encoder_mask1, layer_cache)
        return x


class DecoderlayerTrOrigin(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(DecoderlayerTrOrigin, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)

        self.encoder_decoder_attention_vis = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.drop_vis = nn.Dropout(p=dropout)
        self.feed_forward = FeedForward(d_model)
        self.num_layers = 4
        self.residual_att = ResidualConnection(d_model, dropout=dropout)
        self.drop_last = nn.Dropout(p=dropout)
        self.layerNorm = LayerNorm(d_model)
        self.pad_token_idx = 0

    def forward(self, target, encoder_output, target_mask, encoder_mask1, cache=None):
        x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask, cache))
        x = self.residual_2(x, lambda x: self.encoder_decoder_attention_vis(x, encoder_output, encoder_output,
                                                                            encoder_mask1))
        x = self.residual_att(x, self.feed_forward)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.eps = 10e-9

    def forward(self, x):
        # print(x.shape)
        mean_ = torch.mean(x, -1, keepdim=True)
        var = torch.square(x - mean_).mean(dim=-1, keepdim=True)
        return (x - mean_) / torch.sqrt(var + self.eps)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # print(seq.shape)
    sz_b, len_s, dim_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def get_subsequent_mask2(len_s_):
    ''' For masking out the subsequent info. '''

    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s_, len_s_)), diagonal=1)).bool().cuda()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, max_seq_len, head_num, dropout, train=True, alpha=0.6,
                 beam_size=5, src_pad_idx=0,
                 trg_pad_idx=0, trg_bos_idx=1,
                 trg_eos_idx=2,pre_train_value = 3, vit_encoder = None):
        super(Transformer, self).__init__()
        self.input_drop = nn.Dropout(0.3)
        self.d_model = d_model
        self.preLinear = nn.Linear(2048,d_model,bias=False)
        self.encoders_vis = Encoder(d_model=d_model, head_num=head_num, dropout=dropout, layers_num=num_layers)
        self.decoders = Decoder(d_model=d_model, head_num=head_num, dropout=dropout, layers_num=num_layers)
        self.segment = nn.Embedding(2, d_model).cuda()
        self.num_patches = 225*25
        self.layernorm_enc = LayerNorm(d_model)
        self.layernorm_dec = LayerNorm(d_model)
        self.d_model = d_model
        self.num_heads = head_num
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.matrix_emb = nn.Linear(d_model, vocab_size, bias = False)
        self._reset_parameters()

        self.alpha = alpha
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        self.trg_pad_idx = trg_pad_idx
        self.pre_enc = vit_encoder
        self.pre_train_value = pre_train_value

    def forward(self, vis, target, target_mask, labels=None, epoch=None, eval_=False, vis_mask=None, pre_enc=None,
                predict=False):

        all_lst = self.encoder(vis, None)
        lm_logits = self.decoder(target, all_lst, target_mask, vis_mask)
        loss = None

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return lm_logits, loss

    def encoder(self, vis, vis_mask_sub = None):
        vis_mask = get_subsequent_mask(vis)
        vis = self.preLinear(vis)
        vis_lst = self.encoders_vis(vis, vis_mask)
        return vis_lst

    def decoder(self, target, vis, target_mask, vis_mask, cache=None):
        target = F.embedding(target, self.matrix_emb.weight)
        target = self.decoders(target, vis, target_mask, vis_mask, cache)
        target = self.layernorm_dec(target)
        lm_logits = self.matrix_emb(target)
        return lm_logits

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _get_symbols_to_logits_fn(self, max_decode_length, dec_padding_mask, dec_padding_mask2):
        decoder_self_attention_bias = get_subsequent_mask2(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            with torch.no_grad():
                decoder_input = ids[:, -1:].cuda()

                self_attention_mask = decoder_self_attention_bias[:, int(i.item()):int(i.item()) + 1,
                                      :int(i.item()) + 1].clone().detach()
                dec_output = self.decoder(decoder_input, cache.get("encoder_outputs"), self_attention_mask,
                                          dec_padding_mask, cache).clone().detach()

                logits = dec_output.clone().detach()
                logits = torch.squeeze(logits, dim=1)
                return logits, cache

        return symbols_to_logits_fn

    def predict(self, vis, dec_padding_mask,seg_=None):
        with torch.no_grad():
            batch_size = vis.shape[0]
            max_decode_length = self.max_seq_len
            symbols_to_logits_fn = self._get_symbols_to_logits_fn(
                max_decode_length, dec_padding_mask, seg_)
            initial_ids = torch.LongTensor([self.trg_bos_idx]).requires_grad_(False)
            num_heads = self.num_heads
            dim_per_head = self.d_model // num_heads
            cache = {
                "layer_%d" % layer: {
                    "key":
                        torch.zeros(
                            [batch_size, num_heads, 0, dim_per_head]),
                    "value":
                        torch.zeros(
                            [batch_size, num_heads, 0, dim_per_head])
                } for layer in range(self.num_layers)
            }
            encoder_outputs = self.encoder(vis, None)
            cache["encoder_outputs"] = encoder_outputs
            decoded_ids, scores = make_beam_challenge.sequence_beam_search(
                symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=initial_ids,
                initial_cache=cache,
                vocab_size=self.vocab_size,
                beam_size=self.beam_size,
                alpha=self.alpha,
                max_decode_length=20,
                eos_id=self.trg_eos_idx,
                padded_decode=False,
                dtype="float32")
            top_decoded_ids = decoded_ids[:, 0, 1:]
            top_scores = scores[:, 0]
        return {"outputs": top_decoded_ids, "scores": top_scores}



