import torch

def out_to_image(image):
    return torch.clip(image.int().permute(1, 2, 0), 0, 255)
