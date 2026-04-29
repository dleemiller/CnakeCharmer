"""Image resize, cropping, and feature-match post-processing utilities."""

from __future__ import annotations

import numpy as np


def get_img_dims(img, max_width, max_height, model_input):
    width = img.shape[1]
    height = img.shape[0]

    if width > max_width:
        width = max_width
    elif width % model_input != 0:
        width = (width // model_input + 1) * model_input

    if height > max_height:
        height = max_height
    elif height % model_input != 0:
        height = (height // model_input + 1) * model_input

    return width, height


def cut_img(img, img_cuts_dict):
    cut_bot = img_cuts_dict["cut_bot"] or img.shape[0]
    cut_right = img_cuts_dict["cut_right"] or img.shape[1]
    return img[img_cuts_dict["cut_top"] : cut_bot, img_cuts_dict["cut_left"] : cut_right]


def collect_match_points(matches, points1, points2):
    final_points1 = np.zeros((len(matches), 2), dtype=np.float32)
    final_points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for idx, match in enumerate(matches):
        final_points1[idx, :] = points1[match.queryIdx].pt
        final_points2[idx, :] = points2[match.trainIdx].pt
    return final_points1, final_points2
