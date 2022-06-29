import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import os
import cv2


def create_mask(img_shape, rle_mask):
    """
    Converting RLE-format to mask represented by 4d matrix, each channel represents another class, and the last
    channel represents the background.
    :param img_shape: the shape of the original image (H,W)
    :param rle_mask: List of RLE-segmentation for each class represented by a string (if the class is not in the image
    the element in the list is an empty string '')
    :return img_seg: Mask represented by 4d matrix (H,W,C)
    """
    img_seg = np.zeros((img_shape[0], img_shape[1], 4))
    for i in range(3):
        if not isinstance(rle_mask.iloc[i],
                          str):  # check if the element is empty, no segmentation mask for the current class
            continue
        lis_ezer = [int(i) for i in rle_mask.iloc[i].split(" ")]
        for pos in range(0, len(lis_ezer), 2):
            ezer = np.unravel_index(np.arange(lis_ezer[pos], lis_ezer[pos] + lis_ezer[pos + 1]), img_shape)
            img_seg[ezer[0], ezer[1], i] = 1
    img_seg[:, :, 3] = np.ones((img_shape[0], img_shape[1])) - img_seg[:, :, 0] - img_seg[:, :, 1] - img_seg[:, :, 2]
    img_seg[img_seg < 0] = 0
    return img_seg


def read_image_by_path(path):
    img = mpimg.imread(path)
    return img


def generator_images(df_paths, list_paths):
    """
    :param list_paths: list of the paths of the images
    :param df_paths: Table with the paths of the image and the corresponding segmentation mask in RLE-format.
    :return: The original image, the mask in 4d matrix format, the id of the image
    """
    # filelist = df_paths.path
    for path in list_paths:
        img = read_image_by_path(path)
        id_img = df_paths[df_paths.path == path].id.iloc[0]
        rle_mask = df_paths[df_paths.id == id_img].segmentation
        img_seg = create_mask(img.shape, rle_mask)
        if img_seg[:, :, :3].max() > 0:
            yield (np.expand_dims(img, 2), img_seg, id_img)


def plot_image_mask(img, img_seg, id_img, title="original"):
    fig = plt.figure(figsize=(20, 4))
    num_cols = 5
    plt.suptitle(id_img + "-" + title)
    ax = fig.add_subplot(1, num_cols, 1)
    ax.imshow(img)
    for i in range(4):
        ax = fig.add_subplot(1, num_cols, i + 2)
        ax.imshow(img_seg[:, :, i])
    plt.tight_layout(pad=2)


def generator_batch_images(im_dim, df_paths, batchsize=32, interpolation=cv2.INTER_NEAREST):
    """
    :param im_dim: The new image and mask size (for the resize operation before stack the images in one batch)
    :param df_paths: Table with the paths of the image and the corresponding segmentation mask in RLE-format.
    :param batchsize: (i.e. a batch size of 32 would correspond to 32 images and 32 masks from the generator)
    :return: batch of the original image and batch of the masks in 4d matrix format
    """
    while True:
        ig = generator_images(df_paths, df_paths.dropna().path)
        batch_img, batch_mask = [], []

        for img, mask, id in ig:
            # resize
            mask = cv2.resize(mask, im_dim, interpolation=interpolation)
            img = np.expand_dims(cv2.resize(img, im_dim, interpolation=interpolation), 2)
            # Add the image and mask to the batch
            batch_img.append(img)
            batch_mask.append(mask)
            # If we've reached our batchsize, yield the batch and reset
            if len(batch_img) == batchsize:
                yield np.stack(batch_img, axis=0), np.stack(batch_mask, axis=0)
                batch_img, batch_mask = [], []

        # If we have an nonempty batch left, yield it out and reset
        if len(batch_img) != 0:
            yield np.stack(batch_img, axis=0), np.stack(batch_mask, axis=0)
            batch_img, batch_mask = [], []
