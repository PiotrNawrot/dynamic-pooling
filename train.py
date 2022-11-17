# coding: utf-8

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

import argparse
import functools
import itertools
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
import yaml
import inspect

from torch.nn.parallel import DistributedDataParallel

import neptune.new as neptune
import utils
from data_utils import get_lm_corpus
from hourglass import MemTransformerLM
from utils.exp_utils import create_exp_dir, init_seed, save_checkpoint
from utils.distributed import print_once
from test import autoregressive_test


run = None
np.set_printoptions(suppress=True)


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)

    parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser])

    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default=None)

    config_args, _ = cfg_parser.parse_known_args()

    assert config_args.config is not None and config_args.config_file is not None
    with open(config_args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']

    # Main args
    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='LM-TFM', type=str,
                         help='Directory for the results')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, help='Dataset name')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_head', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=64,
                       help='Head dimension')
    model.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    model.add_argument('--d_inner', type=int, default=2048,
                       help='Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1,
                       help='Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0,
                       help='Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true',
                       help='Apply LayerNorm to the input instead of the output')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--model_config', type=str, default="[3, (8,) ,3]",
                       help="[pre_layers, (shortened_layers, ), post_layers]")
    model.add_argument('--activation_function', type=str, default='relu')

    boundaries = parser.add_argument_group('boundary creator')
    boundaries.add_argument('--boundaries_type', type=str)
    boundaries.add_argument('--tokenizer_path', type=str)
    boundaries.add_argument('--fixed_sf', type=int)
    boundaries.add_argument('--spikes_left', type=int)
    boundaries.add_argument('--temp', type=float)
    boundaries.add_argument('--prior', type=float)

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='adam', type=str, choices=['adam'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.00025,
                     help='Initial learning rate')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine'], help='LR scheduler to use')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam')
    opt.add_argument('--adam_b1', type=float, default=0.9)
    opt.add_argument('--adam_b2', type=float, default=0.999)
    opt.add_argument('--adam_eps', type=float, default=1e-8)

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation')
    training.add_argument('--roll', action='store_true',
                          help='Enable random shifts within each data stream')
    training.add_argument('--shuffle', action='store_true',
                          help='Shuffle text chunks')
    training.add_argument('--fp16', action='store_true', help='Use cuda fp16')
    training.add_argument('--tgt_len', type=int, default=192,
                          help='Number of tokens to predict')
    training.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    training.add_argument('--nw', type=int, default=0,
                          help='Number of workers')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int)
    val.add_argument('--eval_total_len', type=int)
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')
    val.add_argument('--ckpt_path', type=str, default="")

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.ckpt_path = '/'.join(config_args.config_file.split('/')[:-1])
    args.eval_batch_size = args.batch_size * 2

    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    assert args.boundaries_type in [
        "none",
        "fixed",
        "whitespaces",
        "unigram",
        "entropy",
        "gumbel",
    ]

    return args


def evaluate(eval_iter, model, args):
    model.eval()

    stats_agg = defaultdict(list)
    total_len, total_loss = 0, 0.

    with torch.no_grad():
        for i, (data, target, seq_len, boundaries) in enumerate(eval_iter.get_fixlen_iter()):
            if args.eval_max_steps > 0 and i >= args.eval_max_steps:
                break

            data = data.to(eval_iter.device, non_blocking=True)
            data_chunks = torch.chunk(data, args.batch_chunk, 1)

            target = target.to(eval_iter.device, non_blocking=True)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)

            if boundaries is not None:
                boundaries = boundaries.to(eval_iter.device, non_blocking=True)
                boundaries_chunks = torch.chunk(boundaries, args.batch_chunk, 1)
            else:
                boundaries_chunks = None

            for j in range(args.batch_chunk):
                with torch.cuda.amp.autocast(args.fp16):
                    loss, stats, aux_loss, _ = model(
                        data_chunks[j].contiguous(),
                        target_chunks[j].contiguous(),
                        boundaries_gt=boundaries_chunks[j].contiguous()
                        if boundaries_chunks is not None
                        else None,
                    )
                    loss = loss.float().mean().type_as(loss)

                total_loss += seq_len * loss.item()
                total_len += seq_len

            for k, v in stats.items():
                stats_agg[k].append(v)

    model.train()

    return total_loss / total_len, stats_agg


def train_iteration(model, i, data_chunks, target_chunks, boundaries_chunks,
                    args, scaler):
    data_i = data_chunks[i].contiguous()
    target_i = target_chunks[i].contiguous()

    if boundaries_chunks is not None:
        boundaries_i = boundaries_chunks[i].contiguous()
    else:
        boundaries_i = None

    with torch.cuda.amp.autocast(args.fp16):
        seq_loss, stats, aux_loss, _ = model(data_i, target_i, boundaries_i)
        seq_loss = seq_loss.float().mean().type_as(seq_loss)
        total_loss = (seq_loss + aux_loss) / args.batch_chunk

    if args.fp16:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

    return seq_loss.item() / args.batch_chunk, stats


def train(tr_iter, va_iter, model, model_config, optimizer,
          scheduler, vocab, epoch, last_iter, train_step,
          args, scaler):
    model.train()

    train_loss = 0
    target_tokens = 0
    log_step = 0
    log_start_time = time.time()

    # Values that I get in each step and average them out
    # I gather the data only from 1 GPU
    stats_agg = defaultdict(list)

    train_iter = tr_iter.get_fixlen_iter(start=last_iter, shuffle=args.shuffle,
                                         seed=args.seed + epoch, nw=args.nw)

    for batch, (data, target, seq_len, boundaries) in enumerate(train_iter, start=1):
        # Prepare data
        data = data.to(tr_iter.device, non_blocking=True)
        data_chunks = torch.chunk(data, args.batch_chunk, 1)

        target = target.to(tr_iter.device, non_blocking=True)
        target_chunks = torch.chunk(target, args.batch_chunk, 1)

        if boundaries is not None:
            boundaries = boundaries.to(tr_iter.device, non_blocking=True)
            boundaries_chunks = torch.chunk(boundaries, args.batch_chunk, 1)
        else:
            boundaries_chunks = None

        # Update counters
        log_step += 1
        target_tokens += target.numel()

        # Optimizer zero grad
        for param in model.parameters():
            param.grad = None

        # Training on current batch
        for i in range(args.batch_chunk):
            if i < args.batch_chunk - 1 and isinstance(model, DistributedDataParallel):
                with model.no_sync():
                    train_loss_chunk, stats = train_iteration(
                        model, i, data_chunks, target_chunks,
                        boundaries_chunks, args, scaler
                    )
            else:
                train_loss_chunk, stats = train_iteration(
                    model, i, data_chunks, target_chunks, boundaries_chunks,
                    args, scaler
                )

            train_loss += train_loss_chunk

        # Custom stats from the model
        for k, v in stats.items():
            stats_agg[k].append(v)

        if args.fp16:
            scaler.unscale_(optimizer)

        grad_l2 = (
            sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters())
            ** 0.5
        )
        weights_l2 = (
            sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        )

        stats_agg['grad_l2'].append(grad_l2)
        stats_agg['weights_l2'].append(weights_l2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # step-wise learning rate annealing
        train_step += 1

        # linear warmup stage
        if train_step < args.warmup_step:
            curr_lr = args.lr * train_step / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            scheduler.step(train_step - args.warmup_step)

        # logging
        if train_step % args.log_interval == 0 or train_step == 1:
            cur_loss = train_loss / log_step
            cur_loss = utils.distributed.all_reduce_item(cur_loss, op='mean')
            train_loss = 0

            log_step = 0

            lr = optimizer.param_groups[0]['lr']

            elapsed = time.time() - log_start_time
            throughput = target_tokens / elapsed
            throughput = utils.distributed.all_reduce_item(throughput, op='sum')
            target_tokens = 0
            log_start_time = time.time()

            log_str = '| epoch {:3d} step {:>8d} | batches {:>6d} / {:d} | lr {:.3e} ' \
                '| tok/s {:7.0f} | loss {:5.2f}'.format(
                    epoch,
                    train_step,
                    batch,
                    tr_iter.n_batch,
                    lr,
                    throughput,
                    cur_loss,
                )

            print_once(log_str, args)

            if run:
                run['lr'].log(lr, step=train_step)
                run['train/loss'].log(cur_loss, step=train_step)
                run['tokens_per_sec'].log(throughput, step=train_step)
                for k, v in stats_agg.items():
                    run[k].log(np.array(v).mean(), step=train_step)
                stats_agg = defaultdict(list)

        do_periodic_eval = train_step % args.eval_interval == 0
        is_final_step = train_step == args.max_step

        if (do_periodic_eval or is_final_step):
            eval_start_time = time.time()

            val_loss, stats_val = evaluate(va_iter, model, args)
            val_loss = utils.distributed.all_reduce_item(val_loss, op='mean')

            if run:
                run[f"val/loss_tgt{args.eval_tgt_len}_total{args.eval_total_len}"].log(
                    val_loss, step=train_step
                )
                for k, v in stats_val.items():
                    run[f"val/{k}"].log(np.array(v).mean(), step=train_step)

            if run:
                run['val/loss'].log(val_loss, step=train_step)

            print_once('-' * 100, args)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                          train_step // args.eval_interval,
                          train_step,
                          (time.time() - eval_start_time),
                          val_loss,
                          )
            print_once(log_str, args)
            print_once('-' * 100, args)

            last_iter = tr_iter.last_iter

            save_checkpoint(args, model, model_config, optimizer, scheduler,
                            vocab, epoch, batch, last_iter,
                            train_step, args.work_dir, scaler)
            log_start_time += time.time() - eval_start_time

        if is_final_step:
            break

    return train_step


def main():
    args = parse_args()

    # Initialize distributed backend
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.distributed.init_distributed(args.cuda)
    with utils.distributed.sync_workers() as rank:
        if rank == 0:
            create_exp_dir(args.work_dir,
                           scripts_to_save=['train.py', 'hourglass.py'],)

    print_once(f'world size: {utils.distributed.get_world_size()}', args)
    init_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    boundary_kwargs = {
        'boundaries_type': args.boundaries_type,
        'fixed_sf': args.fixed_sf,
        'tokenizer_path': args.tokenizer_path,
    }

    corpus = get_lm_corpus(args.data,
                           args.dataset,
                           **boundary_kwargs)
    vocab = corpus.vocab
    args.n_token = len(vocab)

    tr_iter = corpus.get_iterator(split='train',
                                  bsz=args.batch_size,
                                  tgt_len=args.tgt_len,
                                  device=device,
                                  ext_len=0,
                                  **boundary_kwargs)

    eval_ext_len = args.eval_total_len - args.eval_tgt_len
    va_iter = corpus.get_iterator(split='valid',
                                  bsz=args.eval_batch_size,
                                  tgt_len=args.eval_tgt_len,
                                  device=device,
                                  ext_len=eval_ext_len,
                                  **boundary_kwargs)
    te_iter = corpus.get_iterator(split='test',
                                  bsz=args.eval_batch_size,
                                  tgt_len=args.eval_tgt_len,
                                  device=device,
                                  ext_len=eval_ext_len,
                                  **boundary_kwargs)

    ###########################################################################
    # Build the model
    ###########################################################################
    def get_model_config():
        model_args = inspect.getfullargspec(MemTransformerLM).args
        assert model_args.index('self') == 0
        model_args = model_args[1:]
        return {arg: getattr(args, arg) for arg in model_args}

    # Initialise model
    model = MemTransformerLM(**get_model_config())
    model.apply(functools.partial(utils.weights_init, args=args))
    model.word_emb.apply(functools.partial(utils.weights_init, args=args))
    args.n_all_param = sum([p.nelement() for p in model.parameters()])

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.adam_b1, args.adam_b2),
                           eps=args.adam_eps,
                           weight_decay=args.weight_decay)

    # Scheduler
    max_step = args.max_step
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_step - args.warmup_step, eta_min=0.0)

    # Model to GPU
    model = model.to(device)

    # Autoregressive test
    if args.boundaries_type != 'gumbel':
        # sampling in Gumbel depends on size, so it's hard to implement
        # autoreg test
        with torch.no_grad():
            autoregressive_test(model, device)
            args.autoreg = True

    # Wrap model with DDP
    if torch.distributed.is_initialized():
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        )
    # FP16
    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # Log training and model args
    if rank == 0:
        # Neptune
        global run
        run = neptune.init('syzymon/hourglass-pytorch')
        run['args'] = vars(args)
        run['branch'] = os.getenv('TRAX_BRANCH')
        run['exp_path'] = os.getenv('EXPERIMENT_PATH')
        run['slurm_jobid'] = os.getenv('SLURM_JOB_ID')

    if rank == 0:
        print(model)
        print('=' * 100)
        for k, v in args.__dict__.items():
            print('    - {} : {}'.format(k, v))
        print('=' * 100)

    ###########################################################################
    # Train
    ###########################################################################
    train_step = 0
    for epoch in itertools.count(start=1):
        if args.roll:
            tr_iter.roll(seed=args.seed + epoch)

        train_step = train(
            tr_iter,
            va_iter,
            model,
            get_model_config(),
            optimizer,
            scheduler,
            vocab,
            epoch,
            last_iter=0,
            train_step=train_step,
            args=args,
            scaler=scaler,
        )

        if train_step == args.max_step:
            print('End of training')
            break

    ###########################################################################
    # Test
    ###########################################################################
    test_loss, stats_test = evaluate(te_iter, model, args)
    test_loss = utils.distributed.all_reduce_item(test_loss, op='mean')

    if run:
        run[f'test/loss_tgt{args.eval_tgt_len}_total{args.eval_total_len}'].log(test_loss, step=train_step)
        for k, v in stats_test.items():
            run[f'test/{k}'].log(np.array(v).mean(), step=train_step)

    print_once('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, test_loss / math.log(2)), args
    )

    if run:
        run['test_loss'].log(test_loss, step=train_step)


if __name__ == "__main__":
    main()
