# Efficient Transformers with Dynamic Token Pooling

[**Environment**](#environment) | [**Data**](#data) | [**Training**](#training) | [**Repository**](#repository)

Paper: TODO link

## Environment:

```
pip install -r requirements.txt
```

## Data:
- Download
    - `TODO` 
- Preprocess
    - `TODO` 
- Train Unigram
    - `TODO` 

## Training:
- Training by default starts with a simple test that checks the autoregressive property of a model. We support grad accummulation, distributed training, half precision training.

- To run training use:
```
C=configs/spaces.yaml GPUS= bash scripts/run_exp.sh
```
    - C -> defines the path to the config 
    - GPUS -> defines the number of GPUs for distributed run, when not given then the training runs on a single GPU/CPU

## Repository:

Repository is a fork from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL

We decided to fork from the Nvidia implementation of Transformer XL, because Transformer XL is strong and established baseline in Language Modelling, and Nvidia code is well-optimised for the current hardware.

- ./configs/ 
    - we've prepared configs for all models presented in our work, i.e., Vanilla, Fixed, Entropy, Unigram, Whitespaces, Gumbel
- ./tokenizer_data/ 
    - Pretrained tokenizers using HuggingFace/Sentencepiece library for all datasets we've tested in the paper. You can train them yourself by running:
        - ```python ./tokenizer_data/train_tokenizer.py $ARGS```
        - Args are defined in the `./tokenizer_data/train_tokenizer.py`
- ./cleaners/
    - Implementation of preprocessing rules applied to raw `wiki40b` dataesets and `cc-100` dataset
- Boundary Predictor:
    - {Vanilla, Fixed, Whitespaces}
        - These approaches do not need a boundary predictor. Boundaries are extracted from the data itself in the `boundary_creator.py`, then used in the DataLoader.
    - {Unigram}
        - Segmentation based on Unigram needs a Boundary Predictor, because Unigram itself is not autoregressive. We teach the Boundary Predictor module defined in `hourglass.py` to predict the Unigram segmentation. Boundary Predictor is autoregressive, which makes the whole model autoregressive as well. Unigram segmentation is extracted in `boundary_creator.py`.
    - {Entropy, Gumbel}
        - These approaches are end-to-end and use the main model to train Boundary Predictor. Entire logic is implemented in the `hourglass.py`.


