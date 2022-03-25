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
# from transformers.activations import get_activation
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

        # print(d_model % head_num)
        # assert d_model % head_num != 0 # d_model % head_num == 0 is not, error is occured

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

        # print(d_model % head_num)
        # assert d_model % head_num != 0 # d_model % head_num == 0 이 아닌경우 에러메세지 발생

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
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        batch_num = query.size(0)
        # print(query.shape)
        # print(key.shape)
        # print(query.shape[1])
        k = query.shape[1]
        k1 = key.shape[1]
        k2 = value.shape[1]
        query = self.w_q(query).view(batch_num, k, -1, self.head_num, self.d_k).transpose(2, 3)
        key = self.w_k(key).view(batch_num, k1, -1, self.head_num, self.d_k).transpose(2, 3)
        value = self.w_v(value).view(batch_num, k2, -1, self.head_num, self.d_k).transpose(2, 3)
        # print('query:',query.shape)
        # print('key:', key.shape)
        # print('value:', value.shape)
        attention_result, attention_score = self.self_attention(query, key, value, None)

        # 원래의 모양으로 다시 변형해준다.
        # torch.continuos는 다음행과 열로 이동하기 위한 stride가 변형되어
        # 메모리 연속적으로 바꿔야 한다!
        # 참고 문서: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        attention_result = attention_result.transpose(2, 3).contiguous().view(batch_num, k, -1, self.head_num * self.d_k)
        if attention_result.size(2) == 1:
            attention_result = torch.squeeze(attention_result,dim=2)
        return self.w_o(attention_result)



# timing encoding
def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
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


# sinusoidal positional encoding


'''
class Encoder(nn.Module):
  def __init__(self, d_model, head_num, dropout):
    super(Encoder,self).__init__()
    self.multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward = FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)

  def forward(self, input, mask):
    x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
    x = self.residual_2(x, lambda x: self.feed_forward(x))
    return x
'''


# basic universal transformer
'''
class Encoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, layers_num, frame_length=25, max_length=10000):
        super(Encoder, self).__init__()
        self.layers_num = layers_num
        # self.encoder_layers = nn.ModuleList(
        #    [copy.deepcopy(Encoderlayer(d_model, head_num, dropout)) for _ in range(self.layers_num)]
        # )
        self.encoder_layers = Encoderlayers(d_model, head_num, dropout)
        self.time_encoding = _gen_timing_signal(layers_num, d_model)
        self.positional_encoding = _gen_timing_signal(max_length, d_model)
        # self.segment_encoding = _gen_timing_signal(frame_length, d_model)
        self.segment = nn.Embedding(2,d_model).cuda()
        self.input_drop = nn.Dropout(p=dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, inputs, mask, seg, pos_emb):
        # for enc_layer in self.encoder_layers:
        #    input = enc_layer(input, mask)
        # print(inputs.shape)
        x = self.input_drop(inputs)
        seg_ = seg.view(1, -1)
        # x = inputs
        enc_memory = []
        for _ in range(self.layers_num):
            #  x += self.positional_encoding[:, :inputs.shape[1], :].type_as(inputs.data)

            x += self.time_encoding[:, _, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
            # print(self.segment_encoding[:, :25, :].shape, self.segment_encoding[:, :25, :].unsqueeze(1).shape)

            x += pos_emb[:, :5625]
            x += self.segment(seg_)
            x = self.encoder_layers.forward(x, mask)
            enc_memory.append(self.layernorm(x))
        enc_memory = torch.stack(enc_memory, dim=-1)
        return x, enc_memory
'''
class Encoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, layers_num, max_length=360):
        super(Encoder, self).__init__()
        self.layers_num = layers_num
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(Encoderlayers(d_model, head_num, dropout)) for _ in range(self.layers_num)])
        self.time_encoding = _gen_timing_signal(layers_num, d_model)
        self.positional_encoding = _gen_timing_signal(max_length, d_model)
        self.input_drop = nn.Dropout(p=dropout)
        self.layernorm = LayerNorm(d_model)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def init_weights(self):
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.constant_(self.w.bias, 0)
    def forward(self, inputs, mask):
        x = self.input_drop(inputs)
        # x = inputs
        enc_memory = []
        #enc = []
        for _ in range(self.layers_num):
            x += self.positional_encoding[:, :inputs.shape[1], :].type_as(inputs.data)
            x += self.time_encoding[:, _, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
            x = self.encoder_layers(x, mask)
            # enc_memory.append(self.layernorm(x))
            # if _ != 0:
            #    sig = torch.sigmoid(self.w[_-1](torch.cat([x,enc[_-1]],dim=-1)))
            #    x = x*sig + enc[-1]*(1-sig)
            # enc.append(x)
            # enc_memory.append(self.layernorm(x))
            enc_memory.append(x)

        enc_memory = torch.stack(enc_memory, dim=-2)
        # print(enc_memory.shape)
        return enc_memory


class Encoderlayers(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Encoderlayers, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model)
        # self.feed_forward = FeedForward(d_model, d_ff=2048, dropout=dropout)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)
        #self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, input, mask):
        # vis = self.positional_encoding(vis)
        x = self.residual_1.forward(input, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2.forward(x, lambda x: self.feed_forward(x))
        return x



# basic universal transformer
class Decoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, layers_num, max_length=20):
        super(Decoder, self).__init__()
        self.layers_num = layers_num
        #self.decoder_layers = nn.ModuleList(
        #     [copy.deepcopy(DecoderlayerTVTexpand(d_model, head_num, dropout)) for _ in range(self.layers_num)]
        #)
        self.decoder_layers = DecoderlayerTVTexpand(d_model, head_num, dropout,layers_num)
        self.time_encoding = _gen_timing_signal(layers_num, d_model)
        self.positional_encoding = _gen_timing_signal(max_length, d_model)
        self.input_drop = nn.Dropout(p=dropout)
        #self.w = nn.ModuleList([copy.deepcopy(nn.Linear(d_model * 2, d_model)) for _ in range(self.layers_num)])
        #self.w.apply(self._init_weights)
        # self.w_ = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for i in range(3)])
        # self.w_attention = MultiHeadAttentionStack(d_model=d_model, head_num=head_num)
        # self.drop = nn.Dropout(p=dropout)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, target, encoder_output1, target_mask, encoder_mask1, cache=None):
        # for dec_layer in self.decoder_layers:
        #     target = dec_layer(target, encoder_output1, encoder_output2, target_mask, encoder_mask1, encoder_mask2)
        # target += self.positional_encoding[:, :target.shape[1], :].type_as(target.data)
        x = self.input_drop(target)
        # dec = []
        # print('tar mask : ', target_mask)
        for _ in range(self.layers_num):
            layer_name = "layer_%d" % _

            layer_cache = cache[layer_name] if cache is not None else None
            # no univ
            # x = self.decoder_layers[_](x, encoder_output1, target_mask, encoder_mask1, encoder_mask2, layer_cache)

            # univ
            # if layer_cache is not None and target_mask.size(2)==2:
            #    print(layer_name)
            if cache is not None:
                pos_x = cache[layer_name]['key'].size(2) if cache is not None else 0
                x += self.positional_encoding[:, :target.shape[1]+pos_x, :][:,-1,:].type_as(target.data)
            else:
                x += self.positional_encoding[:, :target.shape[1], :].type_as(target.data)
            x += self.time_encoding[:, _, :].unsqueeze(1).repeat(1, target.shape[1], 1).type_as(target.data)
            x = self.decoder_layers(x, encoder_output1, target_mask, encoder_mask1, encoder_mask1, layer_cache)

        return x

class DecoderlayerTVTexpand(nn.Module):
    def __init__(self, d_model, head_num, dropout, layers_num):
        super(DecoderlayerTVTexpand, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.encoder_decoder_attention_vis = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.drop_vis = nn.Dropout(p=dropout)

        self.drop_motion = nn.Dropout(p=dropout)
        self.feed_forward = FeedForward(d_model)
        self.num_layers = layers_num
        self.pre_attention_layer_v = MultiHeadAttentionStack(d_model=d_model,head_num=head_num)
        self.encoder_decoder_attention_temporal = MultiHeadAttentionStack(d_model=d_model, head_num=head_num)
        self.residual_att = ResidualConnection(d_model, dropout=dropout)
        self.drop_last = nn.Dropout(p=dropout)
        self.layerNorm = LayerNorm(d_model)
        self.pad_token_idx = 0

    def init_weights(self):
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wm.weight)

        nn.init.constant_(self.wv.bias, 0)
        nn.init.constant_(self.wm.bias, 0)

    def forward(self, target, encoder_output, target_mask, encoder_mask1, encoder_mask2, cache=None):

        x = self.residual_1.forward(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask, cache))
        x_ln = self.layerNorm(x)
        encoder_output1 = encoder_output
        x_stack_v = []
        v_ln = self.layerNorm(encoder_output1)
        v_ln_V = self.drop_last(F.relu(self.pre_attention_layer_v.forward(v_ln,
                                                                          v_ln, v_ln, None)))
        #print(v_ln_V.shape)
        v_ln_V += encoder_output1
        #print(v_ln_V.shape)
        for _ in range(self.num_layers):
            # drop 없애는거 실험 필요
            x_vis = self.drop_vis(F.relu(
                self.encoder_decoder_attention_vis(x_ln, self.layerNorm(v_ln_V[:, :, _, :]), self.layerNorm(v_ln_V[:, :, _, :]),
                                                   encoder_mask1)))
            x_stack_v.append(x_vis)

        # print(x_mot_stacked.shape)
        x_vis = torch.stack(x_stack_v,dim=-2)
        # x_motion = torch.stack(x_stack_m,dim=-2)
        x_stacked = torch.stack([x_vis], dim=-2)

        x_last = self.drop_last(F.relu(self.encoder_decoder_attention_temporal(x_ln, x_stacked, x_stacked, None)))
        # print(x_last.shape)
        x_last2 = x + x_last
        x_last3 = self.residual_att(x_last2, self.feed_forward)

        return x_last3


class DecoderlayerTrOrigin(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(DecoderlayerTrOrigin, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)

        self.encoder_decoder_attention_vis = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.drop_vis = nn.Dropout(p=dropout)
        self.feed_forward = FeedForward(d_model)
        # self.test_layer =nn.Linear(d_model,d_model)
        self.num_layers = 4
        self.residual_att = ResidualConnection(d_model, dropout=dropout)
        self.drop_last = nn.Dropout(p=dropout)
        self.layerNorm = LayerNorm(d_model)
        self.pad_token_idx = 0

    def forward(self, target, encoder_output, target_mask, encoder_mask1, cache=None):
        # print(target.shape)
        x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask, cache))
        x = self.residual_2(x, lambda x: self.encoder_decoder_attention_vis(x, encoder_output, encoder_output, encoder_mask1))
        x = self.residual_att(x, self.feed_forward)
        # print(x.shape)
        return x


class MultiHeadAttentionTopk(nn.Module):
    def __init__(self, d_model=512, dropout=0.3,head_num=1,topk=5):
        super(MultiHeadAttentionTopk, self).__init__()

        # print(d_model % head_num)
        # assert d_model % head_num != 0 # d_model % head_num == 0 이 아닌경우 에러메세지 발생
        self.topk = topk
        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num

        self.self_attention_topk = self_attention_topk
        self.dropout = nn.Dropout(p=dropout).cuda()

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        batch_num = query.size(0)

        k = query.shape[1]
        query = query.view(batch_num, k, -1, self.d_k)
        key = key.view(batch_num, k, -1, self.d_k)
        value = value.view(batch_num, k, -1, self.d_k)

        attention_result, attention_score = self.self_attention_topk(query, key, value, None)
        topk_result = torch.topk(attention_score, k=self.topk)

        topk_result = topk_result[1]
        return topk_result


def self_attention_topk(query, key, value, mask=None):
    key_transpose = torch.transpose(key, -2, -1)  # (bath, head_num, d_k, token_)
    matmul_result = torch.matmul(query, key_transpose)  # MatMul(Q,K)
    # query : (b_s, 25, 8, 1, 64)
    # key : (b_s, 25, 8, 2, 64)
    d_k = query.size()[-1]
    attention_score = matmul_result / math.sqrt(d_k)  # Scale

    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e20)

    softmax_attention_score = F.softmax(attention_score, dim=-1)  # 어텐션 값
    result = torch.matmul(softmax_attention_score, value)

    return result, softmax_attention_score


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.eps = 10e-9

    def forward(self, x):
        # print(x.shape)
        mean_ = torch.mean(x, -1, keepdim=True)
        var = torch.square(x - mean_).mean(dim=-1, keepdim=True)
        return (x - mean_) / torch.sqrt(var + self.eps)


class LayerNorm___(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super(LayerNorm___, self).__init__()
        self.ln = nn.LayerNorm(d_model).cuda()

    def forward(self, x):
        return self.ln(x)


def make_std_mask(tg, pad_token_idx):
    target_mask = (tg != pad_token_idx).unsqueeze(-2)
    target_mask = target_mask & Variable(subsequent_mask(tg.size(-1)).type_as(target_mask.data))
    return target_mask.squeeze()

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def get_pad_mask(seq, pad_idx):
    #print()
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def get_subsequent_mask2(len_s_):
    ''' For masking out the subsequent info. '''
    #sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s_, len_s_) ), diagonal=1)).bool().cuda()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, max_seq_len, head_num, dropout, train=True, alpha=0.6,
                 beam_size=5, src_pad_idx=0,
                 trg_pad_idx=0, trg_bos_idx=1,
                 trg_eos_idx=2,pre_train_value = 3,vit_encoder = None):
        super(Transformer, self).__init__()
        self.input_drop = nn.Dropout(0.3)
        self.d_model = d_model
        self.encoders_vis = Encoder(d_model=d_model, head_num=head_num, dropout=dropout, layers_num=num_layers)
        self.decoders = Decoder(d_model=d_model, head_num=head_num, dropout=dropout, layers_num=num_layers)
        self.num_patches = 225*25
        self.layernorm_enc = LayerNorm(d_model)
        self.layernorm_dec = LayerNorm(d_model)
        self.d_model = d_model
        self.num_heads = head_num
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.matrix_emb = nn.Linear(d_model, vocab_size, bias = False)
        self.weight_matrix = nn.Linear(d_model*2,d_model)
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

    def forward(self, vis, target, target_mask, labels=None, epoch = None, eval_ = False, vis_mask = None,pre_enc = None, predict_=False):
        all_lst = self.encoder(vis, vis_mask, epoch, eval_)
        lm_logits = self.decoder(target, all_lst, target_mask, vis_mask)

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return lm_logits, loss

    # def encoder(self, vis, vis_mask, motion, motion_mask):
    def encoder(self, vis,vis_mask, E_, eval_):
        B, T, C, H, W = vis.shape
        vis = rearrange(vis, 'B T C H W -> (B T) C H W')

        if E_ < self.pre_train_value or eval_ == True:
            with torch.no_grad():
                self.pre_enc.eval()
                vis = self.pre_enc(vis,batch_size = B,frames = T)
        else:
            self.pre_enc.train()
            vis = self.pre_enc(vis,batch_size = B,frames = T)

        vis_lst = rearrange(vis[:,0], '(B T) ... -> B T ...', B=B)
        vis_lst2 = rearrange(vis[:,1:], '(B T) ... -> B T ...', B=B)
        shape_2 = vis_lst2.shape[-2]
        frame_data = torch.sum(vis_lst2,dim=-2)/shape_2
        vis_lst = self.layernorm_enc(vis_lst)
        frame_data = self.layernorm_enc(frame_data)
        sig_input = torch.sigmoid(self.weight_matrix(torch.cat([vis_lst,frame_data],dim=-1)))
        final_input = sig_input*vis_lst+(1-sig_input)*frame_data

        final_input *= math.sqrt(self.d_model)
        final_input = self.encoders_vis(final_input, vis_mask)

        return final_input


    def decoder(self, target, vis, target_mask, vis_mask, cache=None):
        target = F.embedding(target, self.matrix_emb.weight)
        target *= math.sqrt(self.d_model)
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

            encoder_outputs = self.encoder(vis, None, 0, True)
            # print(encoder_outputs.shape)
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

            # Get the top sequence for each batch element
            top_decoded_ids = decoded_ids[:, 0, 1:]
            top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}



