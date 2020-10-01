#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import collections
import math
import random
import torch
import time
import pdb
import argparse
import numpy as np
import copy
import sentencepiece as sp
import pickle

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

import torch.multiprocessing
import torch.nn.functional as F

try:
    import nsml
    from nsml import DATASET_PATH
    from nsml import SESSION_NAME
except:
    pass

def main(args, init_distributed=False):

    # NSML NFS Setting

    args.data     = DATASET_PATH + '/FAIR/Data/{}'.format(args.data)
    args.save_dir = DATASET_PATH + '/FAIR/Checkpoints/{}'.format(args.save_dir)
    args.sp_model = args.data + '/sentencepiece.bpe.model'

    ''' Modified '''
    print("| loading tokenizer")
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.Load(args.sp_model)

    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    ''' Modified '''
    src_cands, tgt_cands = get_candidates(epoch_itr, args, tokenizer, task)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr, src_cands, tgt_cands)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr, src_cands, tgt_cands):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        ''' Modified '''
        log_output = trainer.train_step_adv(samples, args, src_cands, tgt_cands, task, epoch_itr.epoch)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument("--num_cands",       default=9,       type=int)
    parser.add_argument("--src_pert_prob",   default=0.33,    type=float)
    parser.add_argument("--tgt_pert_prob",   default=0.33,    type=float)
    parser.add_argument("--sp_model")
    args = options.parse_args_and_arch(parser)
    
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)

'''

We initially get pre-defined number of subword candidates per word (split by whitespace) both in source and target sentences.

Let O be maximal offset number in the batch,

batch_offset : B x O x S, where each subword is assigned to the same number if they belong to the same word
                          for (average) aggregation of wordwise embedding and wordwise gradient

"cand_tokens" and "cand_mask" is for the (average) aggregation of wordwise embeddings

'''


def get_candidates(epoch_itr, args, tokenizer, task):
    
    print('| getting offsets and segment candidates token-wise')
    
    # TODO: Multiprocessing

    src_dictionary = {}
    tgt_dictionary = {}
    
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
        )

    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    for _, batch in enumerate(progress):

        # a bit of hack : 
        # we utilize the fact that elements in the batch are not shuffled
        # we index the first element as the key for indexing batch

        batch_idx   = int(batch['id'][0])
        src_tokens  = batch['net_input']['src_tokens'].numpy()
        tgt_tokens  = batch['target'].numpy() 

        src_dictionary[batch_idx] = get_candidates_batch(args, src_tokens, tokenizer, task.source_dictionary, for_src=True)
        tgt_dictionary[batch_idx] = get_candidates_batch(args, tgt_tokens, tokenizer, task.target_dictionary, for_src=False)


    return src_dictionary, tgt_dictionary


def get_candidates_batch(args, batch_tokens, tokenizer, dictionary, for_src):

    batch_offset    = []
    batch_cands     = []
    max_cand_length = 0
    num_candidates  = args.num_cands

    for i, sent in enumerate(batch_tokens):      
        pad_num    =  sum(sent==dictionary.pad())
        tokens     =  sent[(sent != dictionary.pad()) & (sent != dictionary.eos())]
        text       =  dictionary.string(tokens).split(" ")       
        offset     =  get_offset(text)
        if pad_num != 0:
            pad_offset  = np.array([0]*pad_num)
            if for_src:
                offset  = np.concatenate((pad_offset, offset), axis = None)             
            else:
                offset  = np.concatenate((offset, pad_offset), axis = None)             

        batch_offset.append(np.array(offset))

        text        =  "".join(text).replace("▁", " ").strip().split(" ")
        sent_cands  =  []
                
        for token in text:
            seg_res     = tokenizer.NBestEncodeAsPieces(token, num_candidates) 
            seg_cands   = []
            for cand in seg_res:
                seg_cands.append([dictionary.index(x) for x in cand])

            if len(seg_cands) < num_candidates:
                seg_cands += (num_candidates - len(seg_cands)) * [seg_cands[0]]
                          
            cands_length = len(sum(seg_cands, []))

            if max_cand_length < cands_length:
                max_cand_length = cands_length

            sent_cands.append(seg_cands)
        batch_cands.append(sent_cands)        
    
    batch_offset = np.array(batch_offset) 
    max_offset   = int(batch_offset.max())
    cand_tokens  = np.full((len(batch_tokens), max_offset , max_cand_length), 0) 
    cand_mask    = np.full((len(batch_tokens), max_offset , num_candidates, max_cand_length), 0) 

    for i, sent_cands in enumerate(batch_cands):  
        for j, off_cands in enumerate(sent_cands): 
            s_idx = 0
            for k, seg_cands in enumerate(off_cands):
                cand_mask[i, j, k, s_idx:s_idx+len(seg_cands)] = 1
                s_idx += len(seg_cands)
            
            flattened_cands = sum(off_cands, [])
            cand_tokens[i, j, :len(flattened_cands)] = flattened_cands

    return (batch_offset.astype('int16'), cand_tokens.astype('int16'), cand_mask.astype('int16'), batch_cands)

    # Batch Offset : B x S-1 (non-eos tokens)
    # Cand Tokens  : B x O x C (segmentation candidates are concatenated for each offset)
    # Cand Mask    : B x O x N x C (Mask for each segmentation result for each offset)


def get_offset(str_list):
    
    '''        
    Get offset numpy array
    '''

    offset = []
    count  = 1
    for i, token in enumerate(str_list):
        if "▁" in token and i != 0:
            count += 1
        offset.append(count)

    return np.array(offset)


if __name__ == '__main__':
    cli_main()




