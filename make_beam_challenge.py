# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Beam search to find the translated sequence with the highest probability."""

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import nest_update as nest
import os
import json
import pickle
import torch.nn.functional as F

def inf(dtype):

    if str(dtype) == "torch.float32" or dtype == "bfloat16":
        return 1e7
    elif dtype == "float16":

        return np.finfo(np.float16).max
    else:
        raise AssertionError("Invalid dtype: %s" % dtype)


def type_func(dtype):
    if dtype == "float32":
        return torch.float32


class _StateKeys(object):
    CUR_INDEX = "CUR_INDEX"

    ALIVE_SEQ = "ALIVE_SEQ"
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    ALIVE_CACHE = "ALIVE_CACHE"
    FINISHED_SEQ = "FINISHED_SEQ"
    FINISHED_SCORES = "FINISHED_SCORES"
    FINISHED_FLAGS = "FINISHED_FLAGS"


def _expand_to_same_rank(tensor, target):
    diff_rank = len((target.size())) - len(tensor.size())
    for _ in range(diff_rank):
        tensor = torch.unsqueeze(tensor, -1)
    return tensor


class SequenceBeamSearch(nn.Module):
    def __init__(self,
                 symbols_to_logits_fn,
                 vocab_size,
                 beam_size,
                 alpha,
                 max_decode_length,
                 eos_id,
                 padded_decode,
                 dtype="float32"):

        self.symbols_to_logits_fn = symbols_to_logits_fn
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_decode_length = max_decode_length
        self.eos_id = eos_id
        self.padded_decode = False
        self.dtype = type_func(dtype)

    def search(self, initial_ids, initial_cache):

        batch_size = (
            initial_ids.shape.as_list()[0]
            if self.padded_decode else initial_ids.shape[0])
        state = self._create_initial_state(initial_ids, initial_cache,
                                                         batch_size)

        def _grow_alive_seq(state):

            i = state[_StateKeys.CUR_INDEX]
            alive_seq = state[_StateKeys.ALIVE_SEQ].clone().detach()
            alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS].clone().detach()
            alive_cache = state[_StateKeys.ALIVE_CACHE]

            beams_to_keep = 2 * self.beam_size
            #
            flat_ids = flatten_beam_dim(alive_seq)  # [batch_size * beam_size]

            flat_cache_ = nest.map_structure(flatten_beam_dim, alive_cache)



            #print('flat_cache_', flat_cache_['encoder_outputs'].shape) 
            flat_logits, flat_cache = self.symbols_to_logits_fn(
                flat_ids, i, flat_cache_)

            logits = _unflatten_beam_dim(flat_logits.clone().detach(), batch_size, self.beam_size)
            new_cache = nest.map_structure(
                lambda t: _unflatten_beam_dim(t, batch_size, self.beam_size),
                flat_cache)
            candidate_log_probs = _log_prob_from_logits(logits).cpu().clone().detach()

            log_probs = candidate_log_probs + torch.unsqueeze(alive_log_probs, dim=2)
            flat_log_probs = torch.reshape(log_probs,
                                           [-1, self.beam_size * self.vocab_size])
            topk_log_probs, topk_indices = torch.topk(
                flat_log_probs, k=beams_to_keep, dim=-1)
            topk_beam_indices = topk_indices // self.vocab_size
            topk_seq, new_cache = _gather_beams([alive_seq, new_cache],
                                                topk_beam_indices, batch_size,
                                                beams_to_keep)
            topk_seq = topk_seq.cpu().clone().detach()
            topk_ids = topk_indices % self.vocab_size
            if self.padded_decode:
                topk_seq = torch.transpose(topk_seq, 1, 2).transpose(0, 1)
                topk_seq = torch.index_copy_(topk_seq, [[i + 1]],
                                             torch.unsqueeze(topk_ids, dim=0))
                topk_seq = torch.transpose(topk_seq, 0, 1).transpose(1, 2)
            else:
                topk_seq = torch.cat(
                    [topk_seq, torch.unsqueeze(topk_ids, dim=2)], dim=2)

            return topk_seq, topk_log_probs, topk_ids, new_cache

        def _get_new_alive_state(new_seq, new_log_probs, new_finished_flags,
                                 new_cache):

            new_log_probs1 = new_log_probs.cpu().clone().detach()
            new_finished_flags2 = new_finished_flags.cpu().clone().detach()


            new_log_probs2 = new_log_probs1 + (new_finished_flags2.type(self.dtype) * -inf(self.dtype))

            top_alive_seq, top_alive_log_probs, top_alive_cache = _gather_topk_beams(
                [new_seq, new_log_probs2, new_cache], new_log_probs2, batch_size,
                self.beam_size)


            return {
                _StateKeys.ALIVE_SEQ: top_alive_seq,
                _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
                _StateKeys.ALIVE_CACHE: top_alive_cache
            }

        def _get_new_finished_state(state, new_seq, new_log_probs,
                                    new_finished_flags):

            i = state[_StateKeys.CUR_INDEX]
            finished_seq = state[_StateKeys.FINISHED_SEQ].cpu().clone().detach()
            finished_scores = state[_StateKeys.FINISHED_SCORES].cpu().clone().detach()
            finished_flags = state[_StateKeys.FINISHED_FLAGS].cpu().clone().detach()

            if not self.padded_decode:
                finished_seq0 = torch.cat(
                    [finished_seq,
                     torch.zeros([batch_size, self.beam_size, 1], dtype=torch.int32).cpu()],
                    dim=2).clone().detach()

            length_norm_ = _length_normalization(self.alpha, i + 1, dtype=self.dtype)
            new_scores0 = new_log_probs.clone().detach() / length_norm_

            new_scores_ = new_scores0.clone().detach() + ((1. - new_finished_flags.type(torch.float32)) *
                                                          -inf(self.dtype)).clone().detach()

            finished_seq_ = torch.cat((finished_seq0, new_seq), dim=1)
            finished_scores_ = torch.cat((finished_scores, new_scores_), dim=1)

            finished_flags_ = torch.cat((finished_flags, new_finished_flags), dim=1)
            top_finished_seq, top_finished_scores, top_finished_flags = (
                _gather_topk_beams([finished_seq_, finished_scores_, finished_flags_],
                                   finished_scores_, batch_size, self.beam_size))

            return {
                _StateKeys.FINISHED_SEQ: top_finished_seq,
                _StateKeys.FINISHED_SCORES: top_finished_scores,
                _StateKeys.FINISHED_FLAGS: top_finished_flags
            }

        def _search_step(state):
            new_seq, new_log_probs, topk_ids, new_cache = _grow_alive_seq(state)
            new_finished_flags = torch.eq(topk_ids, self.eos_id)

            alive_state = _get_new_alive_state(new_seq, new_log_probs,
                                               new_finished_flags, new_cache)

            finished_state = _get_new_finished_state(state, new_seq, new_log_probs,
                                                     new_finished_flags)
            new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
            new_state.update(alive_state)
            new_state.update(finished_state)
            return new_state

        with torch.no_grad():

            while self._continue_search(state):

                state = _search_step(state)

        finished_state = state
        return self._process_finished_state(finished_state)

    def _process_finished_state(self, finished_state):
        alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

        finished_seq1 = finished_seq.type(self.dtype)
        alive_seq2 = alive_seq.type(self.dtype)

        finished_seq3 = torch.where(torch.any(finished_flags, 1), finished_seq1, alive_seq2)
        finished_scores3 = torch.where(torch.any(finished_flags, 1), finished_scores, alive_log_probs)

        return finished_seq3, finished_scores3

    def _create_initial_state(self, initial_ids, initial_cache, batch_size):

        cur_index = torch.tensor(0, requires_grad=False)
        alive_seq = expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = torch.unsqueeze(alive_seq, dim=2)

        initial_log_probs = torch.Tensor([[0.] + [-float("inf")] *
                                          (self.beam_size - 1)])


        alive_log_probs = initial_log_probs.repeat([batch_size, 1])

        alive_cache_ = nest.map_structure(
            lambda t: expand_to_beam_size(t, self.beam_size), initial_cache)

        finished_seq = torch.zeros(alive_seq.shape, dtype=torch.float32)

        finished_scores = torch.ones([batch_size, self.beam_size],
                                     dtype=type_func(self.dtype)) * -inf(self.dtype)

        finished_flags = torch.zeros([batch_size, self.beam_size], dtype=torch.bool)

        state = {
            _StateKeys.CUR_INDEX: cur_index,
            _StateKeys.ALIVE_SEQ: alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            _StateKeys.ALIVE_CACHE: alive_cache_,
            _StateKeys.FINISHED_SEQ: finished_seq,
            _StateKeys.FINISHED_SCORES: finished_scores,
            _StateKeys.FINISHED_FLAGS: finished_flags
        }

        return state

    def _continue_search(self, state):

        i = state[_StateKeys.CUR_INDEX]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS].clone().detach()
        finished_scores = state[_StateKeys.FINISHED_SCORES].clone().detach()
        finished_flags = state[_StateKeys.FINISHED_FLAGS].clone().detach()
        not_at_max_decode_length = torch.lt(i, self.max_decode_length)

        max_length_norm2 = _length_normalization(
            self.alpha, self.max_decode_length, dtype=torch.float32)

        best_alive_scores = torch.squeeze(alive_log_probs[:, 0:1],
                                          axis=1) / max_length_norm2

        finished_scores_check = finished_scores.clone().detach() * finished_flags.type(
            torch.float32).clone().detach()

        lowest_finished_scores1 = torch.min(finished_scores_check, axis=1)[0].clone().detach()

        finished_batches = torch.any(finished_flags, 1).clone().detach()

        lowest_finished_scores = lowest_finished_scores1 + ((1.0 - finished_batches.type(torch.float32)) *
                                                            -inf(self.dtype))

        worst_finished_score_better_than_best_alive_score = torch.all(
            torch.greater(lowest_finished_scores.clone().detach(), best_alive_scores.clone().detach()))

        return torch.logical_and(
            not_at_max_decode_length.clone().detach(),
            torch.logical_not(worst_finished_score_better_than_best_alive_score).clone().detach())

def sequence_beam_search(symbols_to_logits_fn,
                         initial_ids,
                         initial_cache,
                         vocab_size,
                         beam_size,
                         alpha,
                         max_decode_length,
                         eos_id,
                         padded_decode=False,
                         dtype="float32"):

    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, beam_size, alpha,
                             max_decode_length, eos_id, padded_decode, dtype)
    return sbs.search(initial_ids, initial_cache)


def _log_prob_from_logits(logits):
    return logits - torch.logsumexp(logits, dim=2, keepdims=True)


def _length_normalization(alpha, length, dtype="float32"):
    return (((5.0 + float(length)) / 6.0) ** alpha)


def expand_to_beam_size(tensor, beam_size):

    tensor = torch.unsqueeze(tensor, dim=1)

    tile_dims = [1] * len(tensor.shape)
    tile_dims[1] = beam_size

    return tensor.repeat(tile_dims)


def flatten_beam_dim(tensor):

    shape = _shape_list(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return torch.reshape(tensor, shape)


def _shape_list(tensor):
    shape = list(tensor.shape)

    dynamic_shape = tensor.shape
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = dynamic_shape[i]
    return shape


def _get_shape_keep_last_dim(tensor):
    shape_list = _shape_list(tensor)

    for i in range(len(shape_list) - 1):
        shape_list[i] = 1

    if isinstance(shape_list[-1], torch.Tensor):
        shape_list[-1] = 1
    return torch.empty(shape_list)


def _unflatten_beam_dim(tensor, batch_size, beam_size):
    shape = _shape_list(tensor)
    new_shape = [batch_size, beam_size] + shape[1:]
    return torch.reshape(tensor, new_shape)


def _gather_beams(nested, beam_indices, batch_size, new_beam_size):
    batch_pos = torch.range(0, batch_size * new_beam_size - 1) // new_beam_size
    batch_pos = torch.reshape(batch_pos, [batch_size, new_beam_size]).cpu().clone().detach()

    coordinates = torch.stack([batch_pos, beam_indices], dim=2)

    return nest.map_structure(lambda state: _gather_nd(state, coordinates),
                              nested)


def _gather_topk_beams(nested, score_or_log_prob, batch_size, beam_size):
    fs = score_or_log_prob.clone().detach()
    for i in range(fs.shape[1]):
        if fs[0][i] < -1e5:
            fs[0][i] -= i
    _, topk_indexes = torch.topk(fs, k=beam_size)

    return _gather_beams(nested, topk_indexes, batch_size, beam_size)


def _gather_nd(params, indices):
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).contiguous().tolist()
    output = params[indices]
    return output.reshape(out_shape).contiguous()