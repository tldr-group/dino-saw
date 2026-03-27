# dino-saw

[![arXiv](https://img.shields.io/badge/arXiv-2508.21529-b31b1b.svg)](https://arxiv.org/abs/2508.21529)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-orange)](https://huggingface.co/rmdocherty/vulture)
[![zenodo](https://img.shields.io/badge/zenodo-10.5281-blue)](https://zenodo.org/records/16993498)

<p align="center">
    <img src="notebooks/resources/summary.jpg">
</p>


The features of Vision Transformers (ViT) like DINOv2 exhibit certain positional biases, including some channels that are highly positional. This is problematic for weakly- or unsupverised downstream tasks that use these features, like k-means clustering or trainable segmentation. 
To combat this, we remove the ViT's learned positional encoding and replace it with 2D aware ALiBi relative distance-based attention offsets. 
This is a pretty destructive action - the positional encoding is injected at the very start of the model - so to recover previous behaviour we finetune this ALiBi-DINOv2 to target the original embeddings.  
This 'retrofit via distillation' works surprisingly well, and produces a model whose features are more homogeneous (i.e. free of positional bias), which improves subsequent segmentations of homogenous electron microscopy data.

## Contents

- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Project Structure](#projectstructure)
- [Citation](#citation)
- [Contact](#contact)
- [References](#references)

## Installation

Download the source and install the package locally:
```bash
git clone https://github.com/tldr-group/dino-saw
pip install . # you could also use uv: uv sync
```

You can also just download `all_in_one_alibi_vit.py` and drop it into your project.

To get the data needed to run all the figures, you need to download and unzip the data from the zenodo to `notebooks/paper_figures/data`.


## Checkpoints

To get the model, you can grab it from hugging face, either by downloading them directly or running this command:

```bash
curl -s -L "https://huggingface.co/rmdocherty/dino-saw/resolve/main/$FILE
```


## Usage

```python
import 

```

## Project structure


TODO:
- cleanup repo
    - type errors?
    - move apply to be an example notebook
- host paper figure data on zenodo
- host models on huggingface
- all in one file?