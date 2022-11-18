# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from shortening import downsample, upsample


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            raise NotImplementedError

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior, bp_type, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds, gt):
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())
        elif self.bp_type in ['gumbel']:
            assert gt is None
            binomial = torch.distributions.binomial.Binomial(
                preds.size(-1),
                probs=torch.Tensor([self.prior]).to(preds.device)
            )
            loss_boundaries = -binomial.log_prob(
                preds.sum(dim=-1)
            ).mean() / preds.size(-1)

            return loss_boundaries

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {
            'acc': acc,
            'precision': precision,
            'recall': recall
        }

        return stats


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, pre_lnorm, model_config,
                 activation_function, boundaries_type, spikes_left,
                 temp, prior,
                 ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )
        self.r_r_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )

        assert pre_lnorm is False, "We didn't use pre_lnorm"

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    activation_function=activation_function)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (shortened_layers, ), post_layers = eval(model_config)

        self.boundaries_type = boundaries_type
        self.is_bp = boundaries_type in ['unigram', 'entropy', 'gumbel']

        if post_layers == 0 and shortened_layers == 0:
            assert boundaries_type == 'none'
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers)
            ])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, d_model).zero_())
            nn.init.normal_(self.null_group)

            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers),
                create_decoder_layers(shortened_layers),
                create_decoder_layers(post_layers),
            ])

            self.down_ln = nn.LayerNorm(d_model)

            # Boundary predictor
            if self.is_bp:
                self.boundary_predictor = BoundaryPredictor(
                    d_model=d_model,
                    d_inner=d_inner,
                    activation_function=activation_function,
                    temp=temp,
                    prior=prior,
                    bp_type=boundaries_type,
                )
                self.spikes_left = spikes_left

        self.final_cast = nn.Linear(d_model, n_token)
        self.crit = torch.nn.CrossEntropyLoss(reduction='none')

    def _forward(self, core_input, layers):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()

        dec_attn_mask = torch.triu(
            core_input.new_ones(qlen, qlen), diagonal=1).bool()

        pos_seq = torch.arange(
            qlen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype
        )

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask
            )

        return core_out

    def get_spikes(self, vector):
        total = torch.ones_like(vector).bool()

        for i in range(1, self.spikes_left + 1, 1):
            mask = vector[i:] > vector[:-i]
            total[i:] &= mask

        return total

    def forward(self,
                data,
                target,
                boundaries_gt):
        """
            data: T x B
            target: T x B
            boundaries_gt: T x B or None
        """
        stats = {}

        # All batches should be of the same length, but last can be shorter
        tgt_len = target.size(0) if target is not None else data.size(0)

        # Token_ids to vector embeddings -> T x B x C
        word_emb = self.word_emb(data)
        hidden = self.drop(word_emb)

        # Extra variables
        loss_boundaries = torch.tensor(0, dtype=data.dtype, device=data.device)
        residual = None

        # Process input with Transformer blocks
        for i in range(len(self.layers)):
            if i == 1:  # Downsampling
                residual = hidden

                if self.boundaries_type in ['fixed', 'whitespaces']:
                    # T x B
                    hard_boundaries = boundaries_gt.float().transpose(0, 1)
                    # B x T
                else:
                    soft_boundaries, hard_boundaries = self.boundary_predictor(hidden)
                    # B x T

                hidden = downsample(
                    boundaries=hard_boundaries,
                    hidden=hidden,
                    null_group=self.null_group,
                )

                hidden = self.down_ln(hidden)

                # Shortening stats
                stats['p_ones'] = (hard_boundaries.sum() / hard_boundaries.numel()).item()
                stats['loss_boundaries'] = loss_boundaries.item()
                stats['shortened_length'] = hidden.size(0)
            elif i == 2:  # Upsampling
                back_hidden = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=hidden,
                )

                hidden = back_hidden + residual

            # Out of downsample / upsample -> regular Transformer blocks
            layers = self.layers[i]

            hidden = self._forward(
                core_input=hidden,
                layers=layers,
            )

        # Calculate loss
        hidden = hidden[-tgt_len:]
        logit = self.final_cast(hidden)

        if self.training or target is not None:
            # T x B x C
            assert hidden.size(0) == target.size(0)

            # Boundary predictor loss
            if self.is_bp:
                if self.boundaries_type == 'entropy':
                    entropy = -torch.nn.functional.log_softmax(
                        logit, dim=-1
                    ) * torch.nn.functional.softmax(logit, dim=-1)

                    entropy = torch.sum(entropy, dim=-1)
                    # T x B

                    target_boundaries = self.get_spikes(entropy).transpose(0, 1)
                    # target_boundaries: B x T
                elif self.boundaries_type in ['unigram']:
                    # T x B
                    target_boundaries = boundaries_gt[-tgt_len:].transpose(0, 1)
                    # B x T
                else:
                    target_boundaries = None

                soft_boundaries = soft_boundaries[:, -tgt_len:]
                hard_boundaries = hard_boundaries[:, -tgt_len:]

                if self.boundaries_type in ['unigram', 'entropy']:
                    assert target_boundaries.sum().item() > 0

                    loss_boundaries = self.boundary_predictor.calc_loss(
                        soft_boundaries, target_boundaries
                    )

                    bp_stats = self.boundary_predictor.calc_stats(
                        hard_boundaries, target_boundaries
                    )

                    for k, v in bp_stats.items():
                        stats[f'{k}'] = v
                elif self.boundaries_type == 'gumbel':
                    loss_boundaries = self.boundary_predictor.calc_loss(
                        preds=hard_boundaries, gt=None
                    )

                    bp_stats = self.boundary_predictor.calc_stats(
                        hard_boundaries, (data == 0).transpose(0, 1)
                    )

                    for k, v in bp_stats.items():
                        stats[f'{k}'] = v

                stats['loss_boundaries'] = loss_boundaries.item()

            # LM loss
            logit = logit.view(-1, logit.size(-1))
            target = target.view(-1)

            loss = self.crit(logit, target)
            loss = loss.view(tgt_len, -1)

            return loss, stats, loss_boundaries, logit
        else:
            # Generation mode, we return raw logits
            return logit
