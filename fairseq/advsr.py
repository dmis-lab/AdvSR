import torch
import torch.nn.functional as F
import copy
import numpy as np
from fairseq import progress_bar

def get_candidates(epoch_itr, args, tokenizer, task):
    
    '''

    We initially acquire pre-defined number of subword candidates per word (split by whitespace) both in source and target sentences for efficiency. (for the cost of memory)

    Let O be maximal of a single word offset in the batch, N be a number of pre-defined number of candidates, C be a maximal number of a single word's candidates' concatenated length

    batch_offset {B x O x S}, where each subword is assigned to the same number if they belong to the same word
                              for (average) aggregation of wordwise embedding and wordwise gradient

    cand tokens  {B x O x C}, segmentation candidates(tokens) are concatenated for each word

    cand mask    {B x O x N x C}, mask for each segmentation result for each offset
    
    '''

    print('| getting offsets and segment candidates token-wise')
    
    # TODO: Multiprocessing

    src_segmentation_cands = {}
    tgt_segmentation_cands = {}
    
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
        )

    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    for _, batch in enumerate(progress):

        # a bit of hack : 
        # we utilize the fact that elements in a batch are not shuffled
        # we index the first element as the key for indexing batch

        batch_idx  = int(batch['id'][0])
        src_tokens = batch['net_input']['src_tokens'].numpy()
        tgt_tokens = batch['target'].numpy() 

        src_segmentation_cands[batch_idx] = get_candidates_batch(args, src_tokens, tokenizer, task.source_dictionary, for_src=True)
        tgt_segmentation_cands[batch_idx] = get_candidates_batch(args, tgt_tokens, tokenizer, task.target_dictionary, for_src=False)

    return src_segmentation_cands, tgt_segmentation_cands


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


def get_offset(str_list):
    
    '''        
    Get word offset of a sentence
    '''

    offset = []
    count  = 1
    for i, token in enumerate(str_list):
        if "▁" in token and i != 0:
            count += 1
        offset.append(count)

    return np.array(offset)
