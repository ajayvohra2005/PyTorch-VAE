import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import torchvision

from dataset import VAEDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, -1,1,1))
# var = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, -1,1,1))

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    
    #x = x * var + mean 
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def make_grid(images, size=256):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


if __name__ == "__main__":
    vae_dataset = VAEDataset(data_path="/home/ubuntu/efs/datasets")
    vae_dataset.setup()

    train_dataloader = vae_dataset.train_dataloader()
    imgs, labels = next(iter(train_dataloader))
    xb = imgs.to(device)[:8]
    grid_img = show_images(xb).resize((8 * 256, 256), resample=Image.NEAREST)
    plt.imshow(grid_img)
    plt.axis('off')  # Turn off axis
    plt.show()
    print("done")

    