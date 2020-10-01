# Adversarial Subword Regualrization for Robust Neural Machine Translation

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
  <img alt="intro" src="https://github.com/JJumSSu/AdvSR/blob/master/img/figure.png" width="500px">
</div>

We utilize gradient signals for exposing diverse, yet adversarial subword sequence for effectively regularizing NMT models.

## Requirements

```bash
$ conda create -n adv_sr python=3.6
$ conda activate adv_sr
$ conda install numpy tqdm nltk
$ pip install sentencepiece
$ conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
```

Once you cloned the github, run

```
$ cd Source_Codes
$ pip install .
```

## Resources

We use the [Fairseq](https://github.com/pytorch/fairseq) (v0.8.0) for training, and [SacreBLEU](https://github.com/mjpost/sacrebleu) for evaluation.

### Datasets

You can run get_data_iwslt15_cs_en.sh for downloading and preprocessing the dataset.

The preprocessed file can be downloaded from the following link.

- [iwslt15.cs.en](https://drive.google.com/open?id=1nqTQba0IcJiXUal7fx3s-KUFRCfMPpaj)

## Train

The following example trains transformer model on IWSLT15.CS.EN dataset.

```bash


```

The 

## Evaluation

The following example evaluates our trained model with IWSLT15.CS.EN on evaluation set.

```bash

```

### Result

The results are as follows.

```

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

