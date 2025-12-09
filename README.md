# dino-saw

Post-hoc homogenisation of DINOv2 features.


## TODO:

1) get it working:
    - try it on DINOv3?
    - minimize final training loss
2) does it improve things?
    - test it on whatever benchmarks DVT (https://arxiv.org/pdf/2401.02957) uses
        - VOC12 segmentation
        - ADE20k segmentation
            (these two are linear probes)
        - object detection w/ frozen features using LOST
            (this uses LOST method - https://github.com/valeoai/LOST/blob/master/object_discovery.py)
        - NYU depth estimation (I imagine ours will be worse)
    - something satellite based, 'DINOv3' - OpenCanopy / segmentation from our features
    - similarity based image recall? 'Decoupling Semantic Similarity from Spatial Alignment for Neural Networks'
    - upsampling + weakly supervised segmentation - R can do this
2) ablations:
    - positional enocdings: no PE, raster (maybe), learned (we have this already), rope, alibi
        - alibi with and without wrapped distance matrix
    - training target for our approach: original embeddings, flipped, translate_featurise
    - linear probing:
        - we can apply this to any trained model (retrained using ours or from literature)
        - examples: normal DINOv2, our finetuned DINOv2, denoising vision transformer's DINOv2, other VITs (ones with alibi encoding like MViT)
        - for series of different images: zeros tensor, dog, satellite image, microstructure (in order of homogeneity)
        - do this for all channels and channelwise
    - finetuning method: from scratch, full finetuune, parameter efficient / LoRA, head network / layer (I imagine this won't work)