# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
import copy
from fairseq import tokenizer
from fairseq.data import data_utils, FairseqDataset, iterators, Dictionary

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FairseqTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, args):
        self.args = args
        self.datasets = {}

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets[split]

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models
        return models.build_model(args, self)

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def get_adv_batch(self, sample, model, criterion, optimizer, args, src_cands, tgt_cands, task, epoch, ignore_grad=False): 
        
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        # we use averaged gradients over batch 
        # TODO : raw gradient?

        src_embeddings_grad   = model.encoder.embed_tokens.weight.grad[sample['net_input']['src_tokens']].detach().clone()
        src_embeddings_weight = model.encoder.embed_tokens.weight.detach().clone()
        tgt_embeddings_grad   = model.decoder.embed_tokens.weight.grad[sample['net_input']['prev_output_tokens']].detach().clone()
        tgt_embeddings_weight = model.decoder.embed_tokens.weight.detach().clone()

        sample_adv = copy.deepcopy(sample)

        sample_adv['net_input']['src_tokens'], sample_adv['net_input']['src_lengths'] = self.get_adv_token(sample, src_cands, src_embeddings_weight, src_embeddings_grad, \
                                                                                                           task.source_dictionary, args, for_src=True)

        sample_adv['net_input']['prev_output_tokens'], sample_adv['target'] = self.get_adv_token(sample, tgt_cands, tgt_embeddings_weight, tgt_embeddings_grad, \
                                                                                                 task.target_dictionary, args, for_src=False)       
        
        return sample_adv
        

    def get_adv_token(self, batch, cands, embeddings_weight, gradients, dictionary, args, for_src):
        
        if for_src:
            embeddings = embeddings_weight[batch['net_input']['src_tokens']] # B x S(pad sent eos) x D
        else:
            embeddings = embeddings_weight[batch['net_input']['prev_output_tokens']] # B x S(eos sent pad) x D

        batch_offset, batch_cands, batch_cands_mask, cands_replace = cands[int(batch['id'][0])] # batch_offset no eos version
        
        ''' Orig Offsetwise Embedding & Gradient Average '''

        batch_offset  = torch.tensor(batch_offset).long().to(device)
        batch_size    = batch_offset.size(0) 
        offset_size   = int(batch_offset.max() + 1) # eos
        seq_size      = batch_offset.size(1)
        mask          = torch.zeros(batch_size, offset_size, seq_size).to(device)
        mask.scatter_(1,batch_offset.unsqueeze(1),1) # B O S // 0 = pad
        eos  = torch.zeros(batch_size, offset_size, 1).to(device)

        if for_src:
            mask = torch.cat([mask, eos], dim = 2) 
        else:
            mask = torch.cat([eos, mask], dim = 2)

        # reverify

        mask  = F.normalize(mask, p=1, dim = 2)
        embedding_offset = torch.bmm(mask, embeddings)[:,1:,:]
        gradient_offset  = torch.bmm(mask, gradients)[:,1:,:]
        
        ''' Cands Offsetwise Embedding Average '''

        batch_cands        = torch.tensor(batch_cands).long().to(device)  
        mask               = torch.tensor(batch_cands_mask).type(torch.FloatTensor).to(device)
        mask               = F.normalize(mask, p=1, dim=3)
        batch_cands_embeds = embeddings_weight[batch_cands]

        mask               = mask.reshape(-1, mask.size(2), mask.size(3)) # BO x N x C
        embeddings         = batch_cands_embeds.reshape(-1, batch_cands_embeds.size(2), batch_cands_embeds.size(3)) # BO x C x 512
        embeddings         = torch.bmm(mask, embeddings) # BO x N x 512
        embeddings         = embeddings.reshape(batch_size, -1, embeddings.size(1), embeddings.size(2))

        ''' Getting Similarity '''

        embeddings = embeddings - embedding_offset.unsqueeze(2) # max (W^T - w')*G
        sim        = F.cosine_similarity(embeddings, gradient_offset.unsqueeze(2), dim=3) # normalizing
        sim        = torch.max(sim, dim=2)[1] 
        sim        = np.array(sim.cpu())

        # today until here

        if for_src:
            adv_batch_idx = []
        else:
            adv_batch_idx = [] 
            adv_tgt_idx   = []

        max_sequence = 0

        for i, sent_idx in enumerate(sim):
            adv_sent_idx = []

            for j, adv_idx in enumerate(sent_idx):
                if j > len(cands_replace[i])-1:
                    continue
                if for_src:
                    if np.random.uniform(0,1) > (1-args.src_pert_prob): # probability
                        adv_sent_idx.append(cands_replace[i][j][adv_idx])
                    else:
                        adv_sent_idx.append(cands_replace[i][j][0])
                if not for_src:
                    if np.random.uniform(0,1) > (1-args.tgt_pert_prob): # probability
                        adv_sent_idx.append(cands_replace[i][j][adv_idx])
                    else:
                        adv_sent_idx.append(cands_replace[i][j][0])

            adv_sent_idx = sum(adv_sent_idx, [])

            if len(adv_sent_idx) > max_sequence:
                max_sequence = len(adv_sent_idx)

            adv_batch_idx.append(np.array(adv_sent_idx))

            if not for_src:
                adv_sent_idx = np.concatenate((adv_sent_idx, np.array([dictionary.eos()])),axis=0)
                adv_tgt_idx.append(adv_sent_idx)

        adv_batch_idx = np.array(adv_batch_idx)

        eos = torch.Tensor(batch_size, 1).fill_(dictionary.eos()).long().to(device)

        if not for_src:
            adv_tgt_idx         = np.array(adv_tgt_idx)
            adv_prev_tgt_padded = list(map(lambda x: self.pad_seq_tgt(x, max_sequence, dictionary), adv_batch_idx))            
            adv_tgt_padded      = list(map(lambda x: self.pad_seq_tgt(x, max_sequence+1, dictionary), adv_tgt_idx))
            adv_prev_tgt_padded = torch.tensor(np.array(adv_prev_tgt_padded)).long().to(device)
            adv_tgt_padded      = torch.tensor(np.array(adv_tgt_padded)).long().to(device)
            adv_prev_tgt_tokens = torch.cat([eos, adv_prev_tgt_padded], dim = 1)

            return adv_prev_tgt_tokens, adv_tgt_padded

        else:
            adv_src_padded = list(map(lambda x: self.pad_seq_src(x, max_sequence, dictionary), adv_batch_idx))
            adv_src_padded = torch.tensor(np.array(adv_src_padded)).long().to(device)
            adv_src_tokens = torch.cat([adv_src_padded, eos], dim = 1)
            adv_src_length = torch.sum(adv_src_tokens!=dictionary.pad(), dim =1)

            return adv_src_tokens, adv_src_length       

    def pad_seq_src(self, sample, max_sequence, dictionary):
        temp = np.full(max_sequence, dictionary.pad())
        temp[max_sequence-len(sample):] = sample
        return temp.astype(int)


    def pad_seq_tgt(self, sample, max_sequence, dictionary):
        temp = np.full(max_sequence, dictionary.pad())
        temp[:len(sample)] = sample
        return temp.astype(int)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def update_step(self, num_updates):
        """Task level update when number of update increases. This is called after optimization step and
           learning rate update of each step"""
        pass

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        return criterion.__class__.aggregate_logging_outputs(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError
