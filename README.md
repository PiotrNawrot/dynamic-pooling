# Efficient Transformers with Dynamic Token Pooling

Paper: TODO link

This repository is a fork from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL. We decided to fork from Nvidia implementation of Transformer XL, because Transformer XL is strong and established baseline in Language Modelling, and Nvidia code is well-optimised for current hardware.

- Environment:
		- Conda

- Data:
		- Preprocessing

- Training:
		- Training by default starts with a simple test that checks autoregressive property of the model
		- To run training use:
			- `C=configs/spaces.yaml GPUS= bash scripts/run_exp.sh`
			- C defines the path to the config 
			- GPUS defines the number of GPUs for distributed run, if not given then it runs on a single GPU/CPU 

- Repository structure:
		- ./configs/ we've prepared configs for all models presented in our work, i.e., Vanilla, Fixed, Entropy, Unigram, Whitespaces, Gumbel

Important files:
- boundary_creator.py
	- There are three different categories of boundaries our model can rely on. They are either created by the BoundaryCreator or SPMBoundaries. SPMBoundaries loads trained SPM tokenizer and extracts boundaries when called by the data loader. BoundaryCreator on the other hand creates boundaries that are somehow predefined like fixed grouping (boundary every 2 elements), random boundaries or boundaries that we can extract from data in an obvious way e.g. whitespaces. It can also return None when initialised with "noboundaries" model - we use it when we rely on boundaries from Boundary Predictor. Boundaries are always created on CPU unless we use Boundary Predictor, it's not an overhead because we have num_workers = 4.
- data_utils.py
	- Implements the data loader and dataset creation. For text files I basically load the stream of characters, pass it to the Counter structure from python to count number of occurences and create a mapping between symbols and integers. Data Loader doesn't use Distributed Sampler, when initialised it splits the dataset based on world size. It either creates boundaries online or when initialised, it uses boundary creator for that.
- hourglass.py & train.py
	- Default training options are fp16. We also support grad accumulation. Both files were taken from Nvidia repo and are very similar. Hourglass's forward pass takes two arguments boundaries_to_use and boundaries_to_predict. boundaries_to_use argument is used whenever we rely on predefined boundaries and we don't have trainable boundary predictor. boundaries_to_predict are used for example when we train boundary predictor to imitate SPM tokenizer. In case we extract boundaries from entropy matrix both arguments are none. 

Less important files / directories:
- cleaners - Implementation of cleaners for text datasets
- tokenizer_data - Pretrained tokenizers from HuggingFace/Sentencepiece library for different datasets/languages.
- utilts - Code copied from Nvidia's Transformer XL repo. It implements some functions for DDP, logging etc. The only important file from utils is vocabulary.py. It implements a simple vocab class that I use to build vocab of the model. It counts the number of occurences of each symbol in the dataset using python's Counter structure and creates a simple mapping between symbols and integers.
