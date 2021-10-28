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
import matplotlib.patches as mpatches

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
print(np.shape(img_series))
img_series = img_series[:, 250:370, 80:200]

# create img for mask building
img = img_series[0]
norm = lambda f, f_min, f_max: (f-f_min)/(f_max - f_min)
vnorm = np.vectorize(norm)
img_norm = filters.rank.median(vnorm(img, np.min(img), np.max(img)), morphology.disk(5)) 

# create mask
markers = exposure.adjust_log(img_norm, 1.2) #  filters.rank.entropy(img_norm, morphology.disk(12))  # 
th_neuron = filters.threshold_otsu(markers)
mask = morphology.closing(markers > th_neuron, morphology.square(1))

# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(mask, cmap='magma')
# plt.show()

# get mask for largest element only
element_label = measure.label(mask)
element_props = measure.regionprops(element_label)
element_area = {element.area : element.label  for element in element_props}
neuron_mask = element_label == element_area[max(element_area.keys())]

# sctive spines detection
der_series = u.series_derivate(img_series,
                               mask=neuron_mask,
                               sigma=2, kernel_size=20,
                               left_w=5, space_w=10, right_w=5)
der_std = ma.masked_where(~neuron_mask, np.std(der_series, axis=0))  # final derivate SD image
th_spine = filters.threshold_minimum(der_std)
spine_mask = der_std > th_spine
spine_label = measure.label(spine_mask)
spine_props = measure.regionprops(spine_label, intensity_image=der_std)
spine_overlay = label2rgb(spine_label, image=markers, bg_label=0, alpha=0.4)


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,10), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img_norm, cmap='magma')
ax[1].imshow(der_std, cmap='jet')
ax[2].imshow(spine_overlay, cmap='nipy_spectral')
# ax[3].plot()

for region in spine_props:
    spine_coord = (region.centroid[1], region.centroid[0])  # spine_mask.shape[0] - 
    print(spine_coord)
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
    ax[0].add_patch(rect)
    # ax[0].scatter(spine_coord[0], spine_coord[1], color='white')
    ax[0].annotate(region.label, xy=(spine_coord[0], minr),  xycoords='data', textcoords='data', color='white',
                   xytext=(spine_coord[0], spine_coord[1]-20),
                   arrowprops=dict(width=2, headwidth=5, headlength=10, facecolor='white', shrink=0.01))

    # spine_mask_ind = spine_mask == region.label
    # spine_series = []
    # for img in img_series:
    #     spine_series.append(np.mean(ma.masked_where(~spine_mask_ind, img)))
    # ax[3].plot(spine_series)

for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
