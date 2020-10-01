## Overview

<h3 align="center">
<p>AdvSR
<a href="https://github.com/dmis-lab/BioSyn/blob/master/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
</h3>
<div align="center">
    <p><b>Adv</b>ersarial <b>S</b>ubword <b>R</b>egularization for Robust Neural Machine Translation 
</div>

<div align="center">
  <img alt="intro" src="https://github.com/JJumSSu/AdvSR/blob/master/img/figure.png" width="400px">
</div>

we present AdvSR to study whether gradient signals during training can be a substitute criterion for choosing segmentation among candidates.
NMT models suffer from typos(character drop, character swap) in the source text due to the unseen subword compositions ( _ denotes segmentation). 
On the other hand, our method correctly decodes them by exposing diverse, yet adversarial subword sequence for effectively regularizing NMT models in low-resource datasets.

## Requirements

```bash
$ conda create -n adv_sr python=3.6
$ conda activate adv_sr
$ conda install numpy tqdm nltk
$ pip install sentencepiece
$ conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
```

Clone & Build

```
$ git clone https://github.com/JJumSSu/AdvSR.git
$ cd AdvSR
$ pip install .
```

## Resources

### Library

We use the [Fairseq](https://github.com/pytorch/fairseq) (v0.8.0) for training, and [SacreBLEU](https://github.com/mjpost/sacrebleu) for evaluation.

### Datasets

You can run ```bash get_data_iwslt15_cs_en.sh``` for downloading and preprocessing the dataset of IWSLT15_CS_EN.
The preprocessed dataset can be downloaded from the following link.

- [IWSLT15_CS_EN](https://drive.google.com/open?id=1nqTQba0IcJiXUal7fx3s-KUFRCfMPpaj)

## Train

The following example trains transformer-base model on IWSLT15_CS_EN.

```bash
CUDA=0
CHECK_DIR=iwslt15.cs.en
DATA_DIR=iwslt15.cs.en
SPM_DIR=iwslt15.cs.en.sentencepiece.sp.model

CUDA_VISIBLE_DEVICES=${CUDA} python train.py ${DATA_DIR} 
                             --max-update 12800 --ddp-backend=no_c10d --arch transformer --optimizer adam --share-decoder-input-output-embed 
                             --adam-betas '(0.9, 0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' 
	                           --warmup-updates 4000 --warmup-init-lr '1e-07'  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy --dropout 0.3 --weight-decay 0.0001 
                             --save-dir ${CHECK_DIR} --max-tokens 4096 --no-epoch-checkpoints 
 		                         --update-freq 8 --num_cands 9 --src_pert_prob 0.25 --tgt_pert_prob 0.25 --adv_sr --sp_model ${SPM_DIR}
```

or

```bash
CUDA=0
CHECK_DIR=iwslt15.cs.en
DATA_DIR=iwslt15.cs.en
SPM_DIR=iwslt15.cs.en.sentencepiece.sp.model

make train CUDA=${CUDA} DATA=${DATA_DIR} CHECK_DIR=${CHECK_DIR} NUM_CANDS=9 SRC_PERT_PROB=0.33 TGT_PERT_PROB=0.33 SP_MODEL=${SPM_DIR}
```

GPU memory will be variable upon training due to the variable length of the adversarially generated sequence.
If OOM occurs(rarely happens), the optimizer will simply skip training the corresponding batch. (as implemented in fairseq)
We experimented with Tesla P40 and we present trained checkpoint from the example.

- [IWSLT15_CS_EN](https://drive.google.com/open?id=1nqTQba0IcJiXUal7fx3s-KUFRCfMPpaj)

## Evaluation

The following example evaluates trained NMT model with the evaluation dataset from IWSLT15_CS_EN.

```bash
make inference CUDA=${CUDA} TEST_DATA=iwslt15/tst2013 SRC=cs TGT=en SP_MODEL=${SPM_DIR} DATA=${DATA_DIR} CHECK_DIR=${CHECK_DIR}/checkpoint_best.pt
```

### Result

The results are as follows.

```bash
BLEU+case.lc+lang.cs-en+numrefs.1+smooth.exp+test.iwslt15/tst2013+tok.13a+version.1.4.2 = 32.5 66.6/40.4/26.5/18.1 (BP = 0.963 ratio = 0.963 hyp_len = 26497 ref_len = 27502)
```

## Citation

```
@inproceedings{park2020adversarial,
  title={Adversarial Subword Regularization for Robust Neural Machine Translation},
  author={Jungsoo Park, Mujeen Sung, Jinhyuk Lee and Jaewoo Kang},
  journal={Proc. of EMNLP Findings},
  year={2020}
  url = {https://arxiv.org/abs/2004.14109}
}
```

