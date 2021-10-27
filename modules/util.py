#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for cell detecting and ROI extraction.
Functions for embrane detection and membrane regions extraction with hysteresis filter.
Optimysed for widefield neuron image.

"""

import os
import logging

import numpy as np
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.external import tifffile
from skimage import filters
from skimage import measure
from skimage import segmentation

from scipy import ndimage as ndi
from scipy.ndimage import measurements as msr


def back_rm(img, edge_lim=20, dim=2):
    """ Background extraction in TIFF series

    For confocal Z-stacks only!
    dem = 2 for one frame, 3 for z-stack

    """
    if dim == 3:
        edge_stack = img[:,:edge_lim,:edge_lim]
        mean_back = np.mean(edge_stack)

        logging.info('Mean background, {} px region: {:.3f}'.format(edge_lim, mean_back))

        img_out = np.copy(img)
        img_out = img_out - mean_back
        img_out[img_out < 0] = 0

        return img_out
    elif dim == 2:
        edge_fragment = img[:edge_lim,:edge_lim]
        mean_back = np.mean(edge_fragment)

        logging.info(f"Mean background, {edge_lim} px region: {mean_back}")

        img = np.copy(img)
        img = img - mean_back
        img[img < 0] = 0

        return img


def de_bleach(img_series, base_num=0):
    """ Wide-field image series bleaching correction.
    Base on first frame apply pixels values correction.

    """
    base_frame = np.mean(img_series[:base_num,:,:])
    corr_series = [frame * (np.sum(frame) / np.sum(base_frame)) for frame in img_series]
    bleach_lvl =  np.sum(img_series[-1,:,:]) / np.sum(img_series[0,:,:])
    logging.info(f'Mean correction deviation={round(bleach_lvl*100, 3)}%') 
    return corr_series


def hyst_mask(img, high=0.8, low=0.2, sigma=3):
    """ Function for neuron region detection with hysteresis threshold algorithm.

    img - input image with higest intensity;
    gen_high - float,  general upper threshold for hysteresis algorithm (percentage of maximum frame intensity);
    sigma - int, sd for gaussian filter.

    Returts cell boolean mask for input frame.

    """
    img_gauss = filters.gaussian(img, sigma=sigma)
    num = 10
    while num > 1:
        mask = filters.apply_hysteresis_threshold(img_gauss,
                                                  low=np.max(img_gauss)*high,
                                                  high=np.max(img_gauss)*low)
        a, num = ndi.label(mask)
        low -= 0.01
    logging.info(f"Lower limit for hyst_mask={round(low, 2)}")
    return mask


def series_derivate(series, mask=False, sigma=4, kernel_size=3,  sd_area=50, sd_tolerance=False, left_w=1, space_w=0, right_w=1, output_path=False):
    """ Calculation of derivative image series (difference between two windows of interes).

    """
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    gauss_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    logging.info(f'Derivate sigma={sigma}')

    der_series = []
    for i in range(np.shape(gauss_series)[0] - (left_w+space_w+right_w)):
        der_frame = np.mean(gauss_series[i+left_w+space_w:i+left_w+space_w+right_w], axis=0) - np.mean(gauss_series[i:i+left_w], axis=0)
        if sd_tolerance:
            der_sd = np.std(der_frame[:sd_area, sd_area])
            der_frame[der_frame > der_sd * sd_tolerance] = 1
            der_frame[der_frame < -der_sd * sd_tolerance] = -1

        der_frame = (der_frame-np.min(der_frame))/(np.max(der_frame)-np.min(der_frame))

        # if not mask:
        #     der_series.append(der_frame)
        # else:
        #
        der_series.append(ma.masked_where(~mask, der_frame))

    logging.info(f'Series len={len(series)}, der. series len={len(der_series)} (left WOI={left_w}, spacer={space_w}, right WOI={right_w})')

    if output_path:
        save_path = f'{output_path}/blue_red'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        norm = lambda x, min_val, max_val: (x-min_val)/(max_val-min_val)  # normilize derivate series values to 0-1 range
        vnorm = np.vectorize(norm)

        for i in range(len(der_series)):
            frame = der_series[i]
            # frame = vnorm(raw_frame, np.min(der_series), np.max(der_series))

            plt.figure(figsize=(20,10))
            ax = plt.subplot()
            img = ax.imshow(frame, cmap='jet')
            img.set_clim(vmin=np.min(der_series), vmax=np.max(der_series)) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.text(10,10,i+1,fontsize=10)
            ax.axis('off')

            plt.savefig(f'{save_path}/frame_{i+1}.png')
            logging.info(f'Derivate frame {i+1} saved!')
            plt.close('all')
    return np.asarray(der_series)


def series_point_delta(series, mask, mask_series=False, baseline_frames=3, delta_min=1, delta_max=-1, sigma=4, kernel_size=5, output_path=False):
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    img_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    baseline_img = np.mean(img_series[:baseline_frames,:,:], axis=0)

    delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0 
    vdelta = np.vectorize(delta)

    if mask_series:
        delta_series = [ma.masked_where(~mask_series[i], vdelta(img_series[i], baseline_img)) for i in range(len(img_series))]
    else:
        delta_series = [ma.masked_where(~mask, vdelta(i, baseline_img)) for i in img_series]

    if output_path:
        save_path = f'{output_path}/delta_F'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(len(delta_series)):
            frame = delta_series[i]

            plt.figure()
            ax = plt.subplot()
            img = ax.imshow(frame, cmap='jet')
            img.set_clim(vmin=delta_min, vmax=delta_max) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.text(10,10,i+1,fontsize=10)
            ax.axis('off')

            file_name = save_path.split('/')[-1]
            plt.savefig(f'{save_path}/{file_name}_frame_{i+1}.png')
            logging.info(f'Delta F frame {i+1} saved!')
            plt.close('all')
        return np.asarray(delta_series)
    else:
        return np.asarray(delta_series)


def alex_delta(series, mask=False, win_index=[5, 10, 15], spacer=0, tolerance=0.03, t_val=200, sigma=False, kernel_size=3, mode='single', min_mask_size=20, output_path=False):
    """ Detecting increasing and decreasing areas, detection limit - sd_tolerance * sd_cell.
    Framses indexes for images calc:

         stimulus
            |
    --------|--------  registration
         |--|--|
         ^  ^  ^   
         |  |  |      
         |  |  win_index[1] + win_index[2] + spacer
         |  win_index[1]
         win_index[1] - win_index[0]

    It's necessary for decreasing of the cell movement influence.
    tolerance - value for low pixel masling, percent from baseline image maximal intensity.

    Mask modes:
    - single - create one mask with all up/down regions
    - multiple - create multiple mask with individual up/down regions, connectivity 8-m, minimal mask size - min_mask_size

    """
    baseline_win = [win_index[1]-win_index[0], win_index[1]]  # frame indexes for baseline image calc
    baseline_img = np.mean(series[baseline_win[0]:baseline_win[1]], axis=0)

    max_win = [win_index[1]+spacer, win_index[1]+win_index[2]+spacer]  # frame indexes for maximal translocations image calc
    max_img = np.mean(series[max_win[0]:max_win[1]], axis=0)

    logging.info(f'Baseline frames indexes:{baseline_win}, max frame indexes:{max_win}')

    if sigma:
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
        baseline_img = filters.gaussian(baseline_img, sigma=sigma, truncate=trun(kernel_size, sigma))
        max_img = filters.gaussian(max_img, sigma=sigma, truncate=trun(kernel_size, sigma))

    # cell_sd = np.std(ma.masked_where(~mask, series[max_frames[0],:,:]))
    # logging.info(f'Cell area SD={round(cell_sd, 2)}')

    diff_img = max_img - baseline_img
    diff_img = diff_img.astype(np.int)  # convert float difference image to integer

    if mask.any():
        delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0  # pixel-wise ΔF/F0
        vdelta = np.vectorize(delta)
        delta_img = ma.masked_where(~mask, vdelta(max_img, baseline_img))

    # up/down mask creating
    t_val = np.max(max_img) * tolerance
    up_mask = np.copy(diff_img) > t_val
    down_mask = np.copy(diff_img) < -t_val

    if mode == 'multiple':
        # label method
        # connectivity = [[1, 1, 1],
        #                 [1, 1, 1],
        #                 [1, 1, 1]]
        # up_label_mask, up_features = ndi.label(up_mask, structure=connectivity)
        # for feature_num in range(0, up_features):
        #     one_up_mask = np.copy(up_label_mask)
        #     one_up_mask[one_up_mask != feature_num] = 0
        #     if np.sum(one_up_mask / feature_num) < min_mask_size:
        #         up_label_mask[up_label_mask == feature_num] = 0

        # gaussian + amplitude filtering method
        up_label_mask = filters.gaussian(up_mask, sigma = 2)
        up_label_mask[up_label_mask < np.max(up_label_mask)*0.2] = 0

    if output_path:
        save_path = f'{output_path}/alex_delta'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if mask.any():
            # px-wise delta F
            plt.figure()
            ax = plt.subplot()
            img = ax.imshow(delta_img, cmap='seismic')
            img.set_clim(vmin=-0.6, vmax=0.6) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}/alex_deltaF.png')
        
        # mean frames
        plt.figure()
        ax0 = plt.subplot(121)
        img0 = ax0.imshow(baseline_img)
        ax0.text(10,10,f'baseline img, frames {baseline_win[0]}-{baseline_win[1]}',fontsize=8)
        ax0.axis('off')
        ax1 = plt.subplot(122)
        img1 = ax1.imshow(max_img)
        ax1.text(10,10,f'max img, frames {max_win[0]}-{max_win[1]}',fontsize=8)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_path}/alex_ctrl.png')
        
        # mean frames diff
        centr = lambda img: abs(np.max(img)) if abs(np.max(img)) > abs(np.min(img)) else abs(np.min(img)) # normalazing vmin/vmax around zero for diff_img

        ax2 = plt.subplot()
        ax2.text(10,10,f'max - baseline',fontsize=8)
        img2 = ax2.imshow(diff_img, cmap='seismic')
        img2.set_clim(vmin=-centr(diff_img), vmax=centr(diff_img))
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img2, cax=cax2)
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_path}/alex_diff.png')

        # up and down masks
        plt.figure()
        ax0 = plt.subplot(121)
        img0 = ax0.imshow(up_mask)
        ax0.text(10,10,'up mask', fontsize=8)
        ax0.axis('off')
        ax1 = plt.subplot(122)
        img1 = ax1.imshow(down_mask)
        ax1.text(10,10,'down mask', fontsize=8)
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_path}/alex_mask.png')

        # up regions labels
        if mode == 'multiple':
            plt.figure()
            ax0 = plt.subplot()
            img0 = ax0.imshow(up_label_mask)
            ax0.text(10,10,'up regions', fontsize=8)
            ax0.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}/alex_up_label.png')

        plt.close('all')
        
        return up_mask, down_mask, baseline_win, max_win
    else:
        return up_mask, down_mask, baseline_win, max_win



if __name__=="__main__":
    pass


# That's all!
