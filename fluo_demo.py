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
from scipy import ndimage as ndi

from skimage import filters
from skimage.morphology import disk
from skimage.segmentation import watershed

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

img = img_series[0]
norm = lambda f, f_min, f_max: (f-f_min)/(f_max - f_min)
vnorm = np.vectorize(norm)
img_norm = filters.rank.median(vnorm(img, np.min(img), np.max(img)), disk(5)) 


markers = filters.rank.entropy(img_norm, disk(12))
th = filters.threshold_otsu(markers)
mask = markers > th

# get largest element mask
element_lab, element_num = ndi.label(mask)
element_area = {np.sum(element_lab[element_lab == i] / i): i for i in range(1, element_num)}
mask = element_lab == element_area[max(element_area.keys())]


der_series = u.series_derivate(img_series,
                               mask=mask,
                               sigma=3, kernel_size=20,
                               left_w=10, space_w=10, right_w=10)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img_norm, cmap='magma')
ax[1].imshow(markers, cmap='magma')
ax[2].imshow(element_lab, cmap='nipy_spectral')
ax[3].imshow(ma.masked_where(~mask, np.std(der_series, axis=0)), cmap='jet')

for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
