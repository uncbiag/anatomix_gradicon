#!/usr/bin/env python
# coding: utf-8



import itk
import anatomix
import matplotlib.pyplot as plt
import unigradicon
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import sys

from monai.inferers import sliding_window_inference

dataset = torch.load("/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/stretched_traintensor/train_imgs_tensor.trch")

print(dataset[0])

sys.exit()





m_anatomix = anatomix.load_model()
def minmax(arr, minclip=None, maxclip=None):
    if not (minclip is None) & (maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
        
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def extract_features(
    img_fixed,
    model,
    fixminclip=None,
    fixmaxclip=None,
    movminclip=None,
    movmaxclip=None,
):
    imfixed = minmax(img_fixed, fixminclip, fixmaxclip)
    imfixed = torch.from_numpy(imfixed)[None, None, ...].float().cuda()
    imfixed.requires_grad = False

    
    with torch.no_grad():
        opfixed = sliding_window_inference(
            imfixed,
            (128, 128, 128),
            2,
            model,
            overlap=0.8,
            mode="gaussian",
            sigma_scale=0.25,
        )
      
    
    return opfixed

sample_image_itk = itk.imread(paths[0])
sample_image = np.array(sample_image_itk)
individual_features = extract_features(sample_image, m_anatomix)
