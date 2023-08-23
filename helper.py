'''
This file contains all the helper functions 
'''

import numpy as np
import cv2
import os 


def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_decode(rle):
    """Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    mask = mask.astype(bool)
    return mask


def see_heatmap_image(heatmap, img, text='Image and heatmap', type=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    heatmap = -heatmap + 1
    mask = heatmap < 0.8
    heatmap = np.uint8(255 * heatmap)
    if type == 0:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_AUTUMN)
    elif type == 1:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_WINTER)
    img[mask] = heatmap[mask]
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def see_points_image(points, img, text='Image and points', point_color=[0,0,255], point_range = 2):
    img_new = img.copy()
    for s in points:
        x = s[1]
        y = s[0]
        for i in range(-point_range,point_range+1):
            for j in range(-point_range, point_range+1):
                if (x+i>=0) and (x+i<img.shape[0]) and (y+j>=0) and (y+j<img.shape[1]):
                    img_new[x+i,y+j] = 0.8*np.array(point_color) + 0.2*img_new[x+i,y+j]
    cv2.imshow(text, img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_new


def see_box_img(box, img, text='Image and box', box_color=[0,0,255], box_width=3):
    img_copy = img.copy()
    for i in range(box_width):
        img_copy[box[1]+i, box[0]:box[2]] = np.array(box_color)
        img_copy[box[3]-i, box[0]:box[2]] = np.array(box_color)
        img_copy[box[1]:box[3], box[0]+i] = np.array(box_color)
        img_copy[box[1]:box[3], box[2]-i] = np.array(box_color)
    cv2.imshow(text, img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def see_mask_image(mask, img, text='SAM', color=[0,0,255]):
    img_copy = img.copy()
    img_copy[mask] = color
    cv2.imshow(text, img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mul(ps, R):
    ones = np.ones((ps.shape[0], 1))
    ps_ = np.concatenate((ps, ones), axis=1)
    re = ps_@R.T
    return re[:,:3]


def relabel(semantic_labels):
    '''
    when load the pth file
    If instance_label == -100, means floor and wall (exclude ceiling)
    If semantic label == -100, means labels no too to consider (eg. school bag, ceilling,, monitor, books...)
    sem label: ('unknown': -100 'wall': 0 , 'floor' : 1, 'cabinet' : 2, 'bed' : 3, 'chair' : 4, 'sofa' : 5, 'table' : 6, 
                'door' : 7, 'window' : 8, 'bookshelf' : 9, 'picture' : 10 , 'counter' : 11, 'desk' : 12,
                'curtain' : 13, 'refrigerator' : 14, 'shower curtain' : 15, 'toilet' : 16, 'sink' : 17, 
                'bathtub' : 18, 'otherfurniture' : 19)
    we switch it to:
    sem label: ('unknown' : -100 'background' : -100, 'cabinet' : 0, 'bed' : 1, 'chair' : 2, 'sofa' : 3, 'table' : 4, 
                'door' : 5, 'window' : 6, 'bookshelf' : 7, 'picture' : 8, 'counter' : 9, 'desk' : 10,
                'curtain' : 11, 'refrigerator' : 12, 'shower curtain' : 13, 'toilet' : 14, 'sink' : 15, 
                'bathtub' : 16, 'otherfurniture' : 17)
    instance_label stays unchanged
    '''
    semantic_labels = np.where(semantic_labels<2, -100, semantic_labels-2)
    return semantic_labels



def relabel2(semantic_labels):
    '''
    when load the pth file
    If instance_label == -100, means floor and wall (exclude ceiling)
    If semantic label == -100, means labels no too to consider (eg. school bag, ceilling,, monitor, books...)
    sem label: ('unknown': -100 'wall': 0 , 'floor' : 1, 'cabinet' : 2, 'bed' : 3, 'chair' : 4, 'sofa' : 5, 'table' : 6, 
                'door' : 7, 'window' : 8, 'bookshelf' : 9, 'picture' : 10 , 'counter' : 11, 'desk' : 12,
                'curtain' : 13, 'refrigerator' : 14, 'shower curtain' : 15, 'toilet' : 16, 'sink' : 17, 
                'bathtub' : 18, 'otherfurniture' : 19)
    we switch it to:
    sem label: ('unknown' : -100 'background' : 0, 'cabinet' : 1, 'bed' : 2, 'chair' : 3, 'sofa' : 4, 'table' : 5, 
                'door' : 6, 'window' : 7, 'bookshelf' : 8, 'picture' : 9, 'counter' : 10, 'desk' : 11,
                'curtain' : 12, 'refrigerator' : 13, 'shower curtain' : 14, 'toilet' : 15, 'sink' : 16, 
                'bathtub' : 17, 'otherfurniture' : 18)
    instance_label stays unchanged
    '''
    semantic_labels = np.where(semantic_labels<1, semantic_labels, semantic_labels-1)
    return semantic_labels


def get_align_matrix(path_txt):
    '''
    Get the rotation matrix to do x-y axis align
    '''
    f_txt = open(path_txt, 'r')
    lines = f_txt.readlines()
    axis_alignment = ''
    for line in lines:
        if line.startswith('axisAlignment'):
            axis_alignment = line
            break
    Rt = np.array([float(v) for v in axis_alignment.split('=')[1].strip().split(' ')]).reshape([4, 4])
    return Rt



def compute_boxes(instance_labels, semantic_labels, coords):
    '''
    compute the foreground instances boxes
    '''
    instances = np.unique(instance_labels)
    box_max_corners = []
    box_min_corners = []
    box_semantics = []
    box_instances = []
    for instance_id in instances:
        if instance_id == -100: # background no boxes 
            continue
        instance_mask = (instance_id == instance_labels)
        instance_sem = semantic_labels[instance_mask][0]
        box_instances.append(instance_id)
        box_semantics.append(instance_sem)
        instance_point_coords = coords[instance_mask]
        max_corner = np.max(instance_point_coords, axis=0)
        min_corner = np.min(instance_point_coords, axis=0)
        box_max_corners.append(max_corner.tolist())
        box_min_corners.append(min_corner.tolist())
    return np.array(box_max_corners), np.array(box_min_corners), np.array(box_semantics, dtype=int), np.array(box_instances, dtype=int)



def add_gaussian_noise_3Dbox(max_corners, min_corners, noise_ratio, box_bounds):
    sigmas = box_bounds * noise_ratio * 0.5
    noise1 = np.random.normal(0, sigmas)
    noise2 = np.random.normal(0, sigmas)
    max_corners_noisy = max_corners + 0.5*noise1
    min_corners_noisy = min_corners - 0.5*noise2
    return max_corners_noisy, min_corners_noisy


def in_box(points, bb_max, bb_min):
    return np.all( points >= bb_min, axis=-1) & np.all( points <= bb_max, axis=-1)


def intersection_volume(min1, max1, min2, max2):
    overlap = np.maximum(0, np.minimum(max1, max2) - np.maximum(min1, min2))
    return np.prod(overlap)


def inclusion_pairs(max_corners, min_corners, bb_volumn, include_rate=0.8):
    pairs = []
    for i in range(len(max_corners)-1):
        for j in range(i+1, len(max_corners)):
            intersect = intersection_volume(min_corners[i], max_corners[i], min_corners[j], max_corners[j])       
            if intersect/bb_volumn[i]>include_rate:
                pairs.append([i,j]) # i is inside j
            elif intersect/bb_volumn[j]>include_rate:
                pairs.append([j,i]) # j is inside i
    return np.array(pairs)


def get_matrix(path_txt):
    '''
    Get the rotation matrix to do x-y axis align
    '''
    f_txt = open(path_txt, 'r')
    lines = f_txt.readlines()
    rows = []
    for line in lines:
        row = np.array([float(v) for v in line.split(' ')])
        rows.append(row)
    return np.stack(rows)


