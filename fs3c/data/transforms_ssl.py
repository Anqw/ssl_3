import logging
import torch
from torch import nn
import numpy as np
import math
from PIL import Image, ImageFilter, ImageOps

import torchvision.transforms as transforms
from . import transform_coord

def normalizer(x):
    mean = [103.530, 116.280, 123.675]
    pixel = [1.0, 1.0, 1.0]
    pixel_mean = torch.Tensor(mean).view(3, 1, 1)
    pixel_std = torch.Tensor(pixel).view(3, 1, 1)
    x = (x - pixel_mean) / pixel_std
    return x

def parse_transform(transform_type):
    method = getattr(transforms, transform_type)
    if transform_type == 'RandomResizedCrop':
        return method(224)
    elif transform_type == 'CenterCrop':
        return method(224)
    elif transform_type == 'Resize':
        return method(224)
    else:
        return method()

def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def con(img):
    image_size = 224
    crop = 0.08
    transform_1 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
        transform_coord.RandomHorizontalFlipCoord(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=1.0),
    ])
    transform_2 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
        transform_coord.RandomHorizontalFlipCoord(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=0.1),
        transforms.RandomApply([ImageOps.solarize], p=0.2),
    ])
    transform = (transform_1, transform_2)
    pil_image = Image.fromarray(img)
    img = transform[0](pil_image)
    img_2 = transform[1](pil_image)
    img, coord = img
    img = normalizer(torch.as_tensor(np.asarray(img).transpose(2, 0, 1).astype("float32")))
    img2, coord2 = img_2
    img2 = normalizer(torch.as_tensor(np.asarray(img2).transpose(2, 0, 1).astype("float32")))
    return img, img2, coord, coord2

def jig(img):
    transform_jigsaw = transforms.Compose([
                    transforms.RandomResizedCrop(255,scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip()])
    transform_patch_jigsaw = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.Lambda(rgb_jittering),])
    permutations = np.load('permutations_35.npy')
    if permutations.min() == 1:
        permutations = permutations - 1
    pil_image = Image.fromarray(img)
    img = transform_jigsaw(pil_image)
    s = float(img.size[0]) / 3
    a = s / 2
    tiles = [None] * 9
    for n in range(9):
        i = int(n / 3)
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a), int(c[0] + a)]).astype(int)
        tile = img.crop(c.tolist())
        tile = transform_patch_jigsaw(tile)
        tile = np.asarray(tile)
        tile = torch.as_tensor(tile.transpose(2, 0, 1).astype("float32"))
        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        tiles[n] = tile

    order = np.random.randint(len(permutations))
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data, 0) # 9*3*H*W
    return data, int(order)

def rot(img):
    transform_list = ['RandomResizedCrop', 'RandomHorizontalFlip']
    transform_funcs = [parse_transform(x) for x in transform_list]
    transform = transforms.Compose(transform_funcs)
    pil_image = Image.fromarray(img)
    rotated_imgs = [
        transform(pil_image),
        transform(pil_image.rotate(90, expand=True)),
        transform(pil_image.rotate(180, expand=True)),
        transform(pil_image.rotate(270, expand=True))
    ]
    rotated_imgs = [np.asarray(x) for x in rotated_imgs]
    rotated_imgs = [torch.as_tensor(x.transpose(2, 0, 1).astype("float32")) for x in rotated_imgs]
    rotated_imgs = [normalizer(x)for x in rotated_imgs]
    rotation_labels = torch.LongTensor([0, 1, 2, 3])
    return torch.stack(rotated_imgs, dim=0), rotation_labels
