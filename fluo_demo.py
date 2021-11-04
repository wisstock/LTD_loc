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

from scipy.ndimage import distance_transform_edt

from skimage import measure
from skimage import filters
from skimage import morphology
from skimage import exposure
from skimage import segmentation
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


img_series = tifffile.imread(f'{data_path}/corr.tif')  # DiffHold_435nm_1
print(np.shape(img_series))
# img_series = img_series[:, 250:370, 80:200]

# create img for mask building
# img = img_series[0]
# norm = lambda f, f_min, f_max: (f-f_min)/(f_max - f_min)
# vnorm = np.vectorize(norm)
img_norm = filters.rank.median(img_series[0], morphology.disk(5)) 

# create mask for nucleus area
nucleus_mask = img_norm < np.max(img_norm)*0.95  # mask bright nucleus area
distances, _ = distance_transform_edt(nucleus_mask, return_indices=True)
expand_nucleus_mask = distances <= 15  # extend nucleus mask

# create mask bright areas
# markers = exposure.adjust_log(img_norm, 1) #  filters.rank.entropy(img_norm, morphology.disk(12))  # 
markers = filters.rank.entropy(img_norm, morphology.disk(8))
th_neuron = filters.threshold_otsu(markers)
mask = morphology.closing(markers > th_neuron, morphology.square(1))

# get mask for largest element only
element_label = measure.label(mask)
element_props = measure.regionprops(element_label)
element_area = {element.area : element.label  for element in element_props}
neuron_mask = element_label == element_area[max(element_area.keys())]
neuron_mask[expand_nucleus_mask] = 0

# sctive spines detection
der_series = u.series_derivate(img_series,
                               mask=neuron_mask,
                               sigma=2, kernel_size=20,
                               left_w=5, space_w=10, right_w=5)
der_std = ma.masked_where(~neuron_mask, np.std(der_series, axis=0))  # final derivate SD image
th_spine = filters.threshold_yen(der_std)
spine_mask = der_std > th_spine

spine_label, spine_num = measure.label(spine_mask, return_num=True)
logging.info(f'{spine_num} spine regions finded')
spine_label = ma.masked_where(~neuron_mask, spine_label)

spine_props = measure.regionprops(spine_label, intensity_image=der_std)
spine_overlay = label2rgb(spine_label, image=img_norm, bg_label=0, alpha=0.4)

# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(spine_label, cmap='jet')
# # ax.imshow(der_std, cmap='magma', alpha=0.5)
# plt.show()




plt.figure(figsize=(20,10))
# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,10), sharex=True, sharey=True)
# ax = axes.ravel()
ax0 = plt.subplot(231)
ax0.imshow(img_norm, cmap='magma')
ax0.axis('off')
ax1 = plt.subplot(232)
ax1.imshow(der_std, cmap='jet')
ax1.axis('off')
ax2 = plt.subplot(233)
ax2.imshow(spine_overlay)
ax2.axis('off')
ax3 = plt.subplot(212)

for region in spine_props:
    spine_coord = (region.centroid[1], region.centroid[0])  # spine_mask.shape[0] - 
    if spine_coord[0] is None:
        logging.warning('Spine region with incorrect properties!')
        continue
    print(spine_coord)
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
    ax0.add_patch(rect)
    # ax[0].scatter(spine_coord[0], spine_coord[1], color='white')
    ax0.annotate(region.label, xy=(spine_coord[0], minr),  xycoords='data', textcoords='data', color='white',
                   xytext=(spine_coord[0], spine_coord[1]-20),
                   arrowprops=dict(width=2, headwidth=5, headlength=10, facecolor='white', shrink=0.01))

    region_mask = spine_label == region.label
    region_mask = region_mask != 0
    spine_series_slice = [np.mean(ma.masked_where(~region_mask, frame)) for frame in img_series]
    ax3.plot(spine_series_slice, label=region.label)

plt.legend()
plt.tight_layout()
plt.show()
