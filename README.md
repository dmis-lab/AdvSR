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

We present AdvSR to study whether gradient signals during training can be a substitute criterion for choosing segmentation among candidates.
NMT models suffer from typos(character drop, character swap) in the source text due to the unseen subword compositions ( _ denotes segmentation). 
Our method correctly decodes them by exposing diverse, yet adversarial subword sequence which effectively regularizes NMT models in the low-resource datasets.

## Requirements

```bash
$ conda create -n adv_sr python=3.6
$ conda activate adv_sr
$ conda install numpy tqdm nltk
$ pip install sentencepiece
$ pip install torch==1.2.0 torchvision==0.4.0
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

The preprocessed dataset can be downloaded from the following link.

- [IWSLT15_CS_EN](https://drive.google.com/drive/folders/1C1Oq4MV0TzoMpsaK0zJHJhgyqhvQVFea?usp=sharing)

Also, you can manually download and preprocess the dataset (IWSLT15.CS.EN) by following example.

```bash
bash prepare_iwslt15_cs_en.sh
RAW_DIR=iwslt15.cs.en.sp16k
DATA_DIR=data-bin/iwslt15.cs.en
make preprocess RAW_DIR=${RAW_DIR} DATA_DIR=${DATA_DIR}
mv ${RAW_DIR}/sentencepiece.sp.model ${DATA_DIR}/.
```

## Train

The following example trains transformer-base model on IWSLT15_CS_EN.

```bash
CUDA=0
CHECK_DIR=iwslt15.cs.en.ckpt
SPM_DIR=${DATA_DIR}/sentencepiece.sp.model
make train_adv CUDA=${CUDA} DATA=${DATA_DIR} CHECK_DIR=${CHECK_DIR} NUM_CANDS=9 SRC_PERT_PROB=0.25 TGT_PERT_PROB=0.25 SPM_DIR=${SPM_DIR}
```

GPU memory will be variable upon training due to the variable length of the adversarially generated sequence.
If OOM occurs(rarely happens), the optimizer will simply skip training the corresponding batch. (as implemented in fairseq)
We experimented with Tesla P40 and present our trained checkpoint from the example.

- [IWSLT15_CS_EN](https://drive.google.com/drive/folders/1C1Oq4MV0TzoMpsaK0zJHJhgyqhvQVFea?usp=sharing)

## Evaluation

The following example evaluates trained NMT model with the evaluation dataset from IWSLT15_CS_EN.
We cloned and updated the codes from [SacreBLEU](https://github.com/mjpost/sacrebleu) for the evaluation of IWSLT15, IWSLT13.

```bash
make inference CUDA=${CUDA} TEST_DATA=iwslt15/tst2013 SRC=cs TGT=en SPM_DIR=${SPM_DIR} DATA=${DATA_DIR} CHECK_DIR=${CHECK_DIR}/checkpoint_best.pt
```

### Result

The result is as follows.

```bash
BLEU+case.lc+lang.cs-en+numrefs.1+smooth.exp+test.iwslt15/tst2013+tok.13a+version.1.4.2 = 32.1 66.5/40.4/26.5/18.1 (BP = 0.954 ratio = 0.955 hyp_len = 26272 ref_len = 27502)
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

