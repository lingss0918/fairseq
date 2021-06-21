# DeCoAr

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Models and code for deep learning representations developed by the AWS AI Speech team:

- [DeCoAR (self-supervised contextual representations for speech recognition)](https://arxiv.org/abs/1912.01679)
- [DeCoAR 2.0 (deep contextualized acoustic representation with vector quantization)](https://arxiv.org/abs/2012.06659)


## Installation

Featurize extraction is done in [Kaldi](https://github.com/kaldi-asr/kaldi).

We expect Python 3.6+. 
```sh
pip install -e .
pip install kaldi_io
```
And `$KALDI_ROOT` should be set.

## Pre-training
We use 80 Filter-bank with feature CMVN. Set the data correct path in config.

```sh
export OMP_NUM_THREADS=1
fairseq-hydra-train
task.data=/home/ubuntu/efs/users/shaosl/decoar2/fairseq/data \
--config-dir examples/decoar/config/pretraining \
--config-name $config
```

For training decoar2, we replaced gumble-softmax with softmax with pre-trained codebook. It leads
to faster convergence and better performance. To pretrain codebook, see config/quant.yaml and then
set codebook path in decoar2.yaml to train decoar2.


## Pre-trained Model
Decoar: https://speech-representation.s3.us-west-2.amazonaws.com/checkpoint_decoar.pt
Decoar2: https://speech-representation.s3.us-west-2.amazonaws.com/checkpoint_decoar2.pt

## References

If you found our package or pre-trained models useful, please cite the relevant work:

**[DeCoAR](https://arxiv.org/abs/1912.01679)**
```
@inproceedings{decoar,
  author    = {Shaoshi Ling and Yuzong Liu and Julian Salazar and Katrin Kirchhoff},
  title     = {Deep Contextualized Acoustic Representations For Semi-Supervised Speech Recognition},
  booktitle = {{ICASSP}},
  pages     = {6429--6433},
  publisher = {{IEEE}},
  year      = {2020}
}
```
**[DeCoAR 2.0](https://arxiv.org/abs/2012.06659)**
```
@misc{ling2020decoar,
      title={DeCoAR 2.0: Deep Contextualized Acoustic Representations with Vector Quantization}, 
      author={Shaoshi Ling and Yuzong Liu},
      year={2020},
      eprint={2012.06659},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
