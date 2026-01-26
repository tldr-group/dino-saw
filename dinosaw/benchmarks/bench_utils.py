import torch


def pascal_colormap():
    cmap = torch.zeros(256, 3, dtype=torch.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = torch.tensor([r, g, b], dtype=torch.uint8)
    return cmap


def colorize(pred_mask):  # pred_mask: (H,W) int
    CMAP = pascal_colormap()
    # returns (3,H,W)
    rgb = CMAP[pred_mask]  # (H,W,3)
    return rgb.permute(2, 0, 1)  # 3,H,W
