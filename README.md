# dino-saw

Post-hoc homogenisation of DINOv2 features.


## TODO:

1) get it working:
    - try interpolating / resacling the alibi distance matrix & train model
        - i.e distance matrix is now relative to image height & width and is rescaled when new (larger) image is passed through
        - i.e always in [0, 1] 
    - try paremeter efficient fine-tuning / a different training schedule (slower + longer)
    - does it need multiscale training?
        - i.e train it at (448, 448) / (518, 58)
        - embed generation will be slow, use flash attention?
    - keep but progressively dropout the original positional encoding during training
        - if above 3 don't work
    - try it on DINOv3?
2) ablations:
    - positional enocdings: no PE, raster (maybe), learned (we have this already), rope, alibi
        - alibi with and without wrapped distance matrix
    - linear probing:
        - we can apply this to any trained model (retrained using ours or from literature)
        - examples: normal DINOv2, our finetuned DINOv2, denoising vision transformer's DINOv2, other VITs (ones with alibi encoding like MViT)
        - for series of different images: zeros tensor, dog, satellite image, microstructure (in order of homogeneity)
        - do this for all channels and channelwise
    - training target for our approach: original embeddings, flipped, translate_featurise
    - finetuning method: from scratch, full finetuune, parameter efficient / LoRA, head network / layer (I imagine this won't work)
3) does it improve things?
    - test it on whatever benchmarks DVT uses
        - VOC12 segmentation
        - ADE20k
        - NYU depth estimation (I imagine ours will be worse)
        - object detection w/ frozen features using LOST
    - similarity based image recall?
    - upsampling + weakly supervised segmentation
    - something else microstructure based?
    