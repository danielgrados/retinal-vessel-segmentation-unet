#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:48:13 2026

@author: danielgp
"""

# dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class AplicarCLAHE:
    """Aplica CLAHE al canal verde de imágenes de fondo de ojo"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)

        if len(img_np.shape) == 2:
            return img

        canal_verde = img_np[:, :, 1]
        verde_mejorado = self.clahe.apply(canal_verde)

        img_np[:, :, 0] = verde_mejorado
        img_np[:, :, 1] = verde_mejorado
        img_np[:, :, 2] = verde_mejorado

        return Image.fromarray(img_np)

class DRIVEDataset(Dataset):
    def __init__(self, root_dir, transform_img=None, transform_mask=None):
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        images_path = os.path.join(root_dir, 'training', 'images')
        masks_path = os.path.join(root_dir, 'training', '1st_manual')

        self.images = sorted([f for f in os.listdir(images_path) if f.endswith('.tif')])
        self.masks = sorted([f for f in os.listdir(masks_path) if f.endswith('.gif')])

        self.image_dir = images_path
        self.mask_dir = masks_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()

        return image, mask

class ChaseDataset(Dataset):
    def __init__(self, root_dir, transform_img=None, transform_mask=None):
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        all_files = os.listdir(root_dir)
        self.image_dir = root_dir
        self.mask_dir = root_dir

        self.images = sorted([f for f in all_files if f.endswith('.jpg')])

        self.masks = []
        for img in self.images:
            base_name = img.replace('.jpg', '')
            mask_candidate = f"{base_name}_1stHO.png"
            if mask_candidate in all_files:
                self.masks.append(mask_candidate)
            else:
                possible = [f for f in all_files if f.startswith(base_name) and f.endswith('.png')]
                self.masks.append(possible[0] if possible else None)

        valid = [(img, msk) for img, msk in zip(self.images, self.masks) if msk]
        self.images, self.masks = zip(*valid) if valid else ([], [])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()

        return image, mask