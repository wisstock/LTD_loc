 #!/usr/bin/env python3
""" Copyright © 2021 Borys Olifirov
LA data analysis.

"""

import sys
import os
import logging

import numpy as np
import numpy.ma as ma
import pandas as pd

from skimage import measure
from skimage import filters
from skimage import morphology
from skimage import exposure
from skimage.color import label2rgb

import tifffile

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('modules')
import util as u


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


data_path = os.path.join(sys.path[0], 'data')
res_path = os.path.join(sys.path[0], 'results')

if not os.path.exists(res_path):
    os.makedirs(res_path)


img_series = tifffile.imread(f'{data_path}/DiffHold_435nm_1.tif')

# create img for mask building
img = img_series[0]
norm = lambda f, f_min, f_max: (f-f_min)/(f_max - f_min)
vnorm = np.vectorize(norm)
img_norm = filters.rank.median(vnorm(img, np.min(img), np.max(img)), morphology.disk(5)) 

# create mask
markers = exposure.adjust_log(img_norm, 1.4) #  filters.rank.entropy(img_norm, morphology.disk(12))
th_neuron = filters.threshold_otsu(markers)
mask = morphology.closing(markers > th_neuron, morphology.square(3))

# get mask for largest element only
element_lab, element_num = measure.label(mask, return_num=True)
element_area = {np.sum(element_lab[element_lab == i] / i): i for i in range(1, element_num)}
neuron_mask = element_lab == element_area[max(element_area.keys())]

# sctive spines detection
der_series = u.series_derivate(img_series,
                               mask=neuron_mask,
                               sigma=2, kernel_size=20,
                               left_w=5, space_w=10, right_w=5)
der_std = ma.masked_where(~neuron_mask, np.std(der_series, axis=0))
th_spine = filters.threshold_minimum(der_std)
spine_mask = der_std > th_spine

spine_overlay = label2rgb(measure.label(spine_mask), image=markers, bg_label=0)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img_norm, cmap='magma')
ax[1].imshow(element_lab, cmap='nipy_spectral')
ax[2].imshow(der_std, cmap='jet')
ax[3].imshow(spine_overlay, cmap='nipy_spectral')

for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
