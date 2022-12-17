from tabnanny import check

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from copy import copy
from matplotlib.patches import Rectangle
from scipy import ndimage
import math
from torchvision.transforms import Resize as tResize, ToPILImage as tToPILImage, ToTensor as tToTensor, \
    Compose as tCompose
import cv2
from datetime import datetime


def rot_img_cv(img, deg):
    # grab the dimensions of the image and calculate the center of the
    # image
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by deg degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h))
    return img_rot


def mat_mul(A, B):
    # cast into <a href="https://geekflare.com/numpy-reshape-arrays-in-python/">NumPy array</a> using np.array()
    return np.array([[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A])


def create_rot_matrix(deg):
    rad = math.radians(deg)
    print(f"deg({deg}) = rad({rad})")
    return [[np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]]


def convert_points_3col(pt_list):
    one_list = np.ones((np.shape(pt_list)[0], 1))
    # print(f"one_list shape = {np.shape(one_list)}")
    # print(f"pt_list shape = {np.shape(pt_list)}")
    return np.c_[np.array(pt_list), one_list]


def plot_circle_given_center_point():
    return


def get_indices_of_image(w, h):
    # from itertools import product
    # def get_indices_of_image(w, h):
    #    xs = range(w)
    #    ys = range(h)
    #    return np.array(list(product(xs, ys)))
    xs = np.tile(np.array([np.arange(w)]), (h, 1))
    ys = np.tile(np.array([np.arange(h)]), (w, 1)).transpose()
    return np.c_[ys.ravel(), xs.ravel()]


def rotate_pixels(idx, rm):
    return mat_mul(idx, rm)


def get_rotated_pixel_vals(im_w, im_h, rot_c, rot_deg):
    rm = create_rot_matrix(rot_deg)
    pix_vals_original, center_add = img_rotator._get_pixels_of_image_given_center(w=im_w, h=im_h, c=rot_c)
    rot_pixels = rotate_pixels(pix_vals_original, rm)
    return rot_pixels, pix_vals_original, center_add


def rotate_img(img, degree):
    rotated_img = ndimage.rotate(img, degree)
    return rotated_img


def get_rectangle(block_corner, block_wh):
    return Rectangle(block_corner, block_wh["w"], block_wh["h"], linewidth=3, edgecolor='r', facecolor='none')


def draw_box(img, r):
    plt.imshow(img, origin="upper")
    plt.gca().add_patch(r)
    plt.show()


def make_uint8_bw(img):
    o = img.astype(np.float32)
    mv = np.mean(np.mean(o))
    x1, x2 = np.argwhere(o < mv), np.argwhere(o >= mv)
    o[x1[:, 0], x1[:, 1]] = 0.0
    o[x2[:, 0], x2[:, 1]] = 1.0
    return o.astype(np.uint8)


def make_bw(img):
    o = img.astype(np.float32)
    mv = np.mean(np.mean(o))
    x1, x2 = np.argwhere(o < mv), np.argwhere(o >= mv)
    o[x1[:, 0], x1[:, 1]] = -1.0
    o[x2[:, 0], x2[:, 1]] = 1.0
    return o


def crop_rect_from_img(img, r):
    box = {
        "col_w_beg": r.xy[0],
        "row_h_beg": r.xy[1],
        "col_w_end": r.xy[0] + r.get_width(),
        "row_h_end": r.xy[1] + r.get_height(),
        "col_width": r.get_width(),
        "row_height": r.get_height(),
    }
    # rows = np.arange(box["row_h_beg"], box["row_h_end"])
    # cols = np.arange(box["col_w_beg"], box["col_w_end"])
    # print("rows : ", rows)
    # print("cols : ", cols)
    if len(np.shape(img)) == 2:
        return img[box["row_h_beg"]:box["row_h_end"], box["col_w_beg"]:box["col_w_end"]].copy()
    elif len(np.shape(img)) == 3:
        return img[box["row_h_beg"]:box["row_h_end"], box["col_w_beg"]:box["col_w_end"], :].copy()
    assert False, f"dimension of image must be either 2 or 3 not {len(np.shape(img))}"


# now try to get the same crop from the rotated - img_rot
def get_rotated_box(img_shape, corner, wh, rot_deg):
    rm = create_rot_matrix(rot_deg)
    # box_corners_original
    f_bco = np.asarray(
        [[corner["col_w"] + 0, corner["row_h"] + 0, 1.0],
         [corner["col_w"] + wh["w"], corner["row_h"] + 0, 1.0],
         [corner["col_w"] + wh["w"], corner["row_h"] + wh["h"], 1.0],
         [corner["col_w"] + 0, corner["row_h"] + wh["h"], 1.0], ], dtype=float)
    # rotate center
    rot_c = [img_shape[1] / 2, img_shape[0] / 2, 1.0]
    # box_corners_centered
    f_bcc = f_bco - rot_c + np.asarray([0, 0, 1], dtype=float)
    rot_pixels = rotate_pixels(f_bcc, rm) + rot_c - np.asarray([0, 0, 1], dtype=float)
    # return rot_pixels, pix_vals_original, center_add
    print(f"rm ({rm}) ")
    print(f"img_shape ({img_shape}) ")
    print(f"corner ({corner}) ")
    print(f"wh ({wh}) ")
    print(f"f_bco\n({f_bco}) ")
    print(f"rot_c\n({rot_c}) ")
    print(f"f_bcc\n({f_bcc}) ")
    print(f"rot_pixels\n({rot_pixels}) ")
    return f_bcc, rot_c, rot_pixels, rm


def crop_rect_for_bw_search(img, r):
    return crop_rect_from_img(make_uint8_bw(img), r)


def rect_to_corners(r):
    w, h = r.get_width(), r.get_height()
    corners = np.asarray(
        [[r.xy[0] + 0, r.xy[1] + 0, 1.0],
         [r.xy[0] + w, r.xy[1] + 0, 1.0],
         [r.xy[0] + w, r.xy[1] + h, 1.0],
         [r.xy[0] + 0, r.xy[1] + h, 1.0]], dtype=float)
    return corners


def crop_and_show(img, r, figsize=(20, 10), title_color="red"):
    img_box = crop_rect_from_img(img, r)
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(img)
    corners = rect_to_corners(r)
    for i in range(4):
        ax[0].plot([corners[i, 0], corners[(i + 1) % 4, 0]], [corners[i, 1], corners[(i + 1) % 4, 1]], color="black",
                   linewidth=3)
    ax[0].set_title(f"image of shape {img.shape}\n {r}", color=title_color)
    ax[1].imshow(img_box)
    ax[1].set_title(f"crop shape {img_box.shape}", color=title_color)
    plt.show()
    return img_box


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float32)


def resize_img(img, perc=None, max_w=None, max_h=None):
    h, w = img.shape[0], img.shape[1]
    if perc is None and max_w is not None and w > max_w:
        perc = max_w / w
    elif perc is None and max_h is not None and h > max_h:
        perc = max_w / w
    elif perc is None:
        perc = 1.0
    w_new = int(w * perc)
    h_new = int(h * perc)
    if perc < 1.0:
        print(f"resizing {w}x{h} to {w_new}x{h_new}")
    t = tCompose([tToTensor(), tToPILImage(), tResize([h_new, w_new])])
    img_resized = np.array(t(img))
    return img_resized, perc


def search_iminim(img, img_box):
    if len(np.shape(img)) == 3:
        img = rgb2gray(img)
    if len(np.shape(img_box)) == 3:
        img_box = rgb2gray(img_box)
    print(f"will convolve im({img.shape}) with cf({img_box.shape})")

    img, perc = resize_img(img.astype(np.float32), perc=None, max_w=1024)
    img_bw = make_bw(img)
    img_box, _ = resize_img(img_box.astype(np.float32), perc=perc)
    img_box_bw = make_bw(img_box)

    print(f"perc({perc})")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kern_sz = np.shape(img_box_bw)
    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kern_sz, stride=1, padding=0)
    conv.weight[0, 0, :, :] = torch.from_numpy(img_box_bw)
    # conv.weight[1, 0, :, :] = torch.from_numpy(1 - img_box)

    conv.to(device)
    img_1 = torch.from_numpy(np.asarray(img_bw, dtype=np.float32)).to(device)
    # img_2 = torch.from_numpy(np.asarray(1 - img_bw, dtype=np.float32)).to(device)
    y1 = conv(img_1.view(1, 1, img_1.shape[0], img_1.shape[1]))
    # ·y2 = conv(img_2.view(1, 1, img_2.shape[0], img_2.shape[1]))
    res_img_1 = y1.cpu().detach().numpy().squeeze()
    # res_img_2 = y2.cpu().detach().numpy().squeeze()
    res_find = res_img_1  # [0, :, :]  - res_img_1[1, :, :] + res_img_2[1, :, :] - res_img_2[0, :, :]
    print(f"will find in res_find({res_find.shape})")
    rh, cw = np.unravel_index(res_find.argmax(), res_find.shape)
    print(f"expected max value is {img_box.size}\n",
          f"faulty max can be {np.sum(img_box_bw.ravel())}\n",
          f"max val is {res_find[rh, cw]}")
    block_corner = {"row_h": rh, "col_w": cw}
    block_wh = {"w": img_box.shape[1], "h": img_box.shape[0]}
    r = get_rectangle((block_corner["col_w"], block_corner["row_h"]), block_wh)
    print(f"rect approximately at xcw({cw / perc}), yrh({rh / perc})")
    return r, img, img_box, res_find


def display_iminim_results(ret_dict, step, title_str="", imgname=""):
    plt.clf()
    if step == 0:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(ret_dict["img"])
        new_r = copy(ret_dict["rect_crop"])
        new_r.set(edgecolor="r")
        ax[0].add_patch(new_r)
        ax[1].imshow(ret_dict["img_box"])
        ax[0].set_title(f"image input with size {ret_dict['img'].shape}")
        ax[1].set_title(f"r({new_r})\nbox size {ret_dict['img_box'].shape}")
        fig.suptitle(title_str)
        plt.savefig("step0" + imgname + ".jpeg")
        plt.show()
    if step == 1:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(ret_dict["rs_img"], cmap="gray")
        new_r = copy(ret_dict["rect_crop_new"])
        new_r.set(edgecolor="white")
        ax[0].add_patch(new_r)
        ax[1].imshow(ret_dict["rs_img_box"], cmap="gray")
        ax[0].set_title(f"image input resized {ret_dict['rs_img'].shape}")
        ax[1].set_title(f"r({new_r})\nbox resized {ret_dict['rs_img_box'].shape}")
        fig.suptitle(title_str)
        plt.savefig("step1" + imgname + ".jpeg")
        plt.show()
    if step == 2:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(ret_dict["s_rs_img"], cmap="gray")
        new_r = copy(ret_dict["rect_crop_new"])
        new_r.set(edgecolor="orange")
        ax[0].add_patch(new_r)
        ax[1].imshow(ret_dict["s_rs_img_box"], cmap="gray")
        ax[0].set_title(f"image -+ wSize {ret_dict['s_rs_img'].shape}")
        ax[1].set_title(f"r({new_r})\nbox -+ wSize {ret_dict['s_rs_img_box'].shape}")
        fig.suptitle(title_str)
        plt.savefig("step2" + imgname + ".jpeg")
        plt.show()
    if step == 3:
        plt.figure(figsize=(20, 10))
        plt.imshow(ret_dict["result_img"], cmap="gray")
        new_r = copy(ret_dict["result_rect"])
        new_r.set(edgecolor="black")
        plt.gca().add_patch(new_r)
        plt.title(f"image wSize {ret_dict['result_img'].shape}")
        plt.savefig("step3" + imgname + ".jpeg")
        plt.show()
    if step == 4:
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].imshow(ret_dict["res_find_1"], cmap="gray")
        ax[1].imshow(ret_dict["res_find_2"], cmap="gray")
        ax[2].imshow(ret_dict["res_find_f"], cmap="gray")
        ax[0].set_title("a=im_conv")
        ax[1].set_title("b=ones_conv")
        ax[2].set_title("a / b")
        for i in range(3):
            new_r = copy(ret_dict["rect_crop_modif"])
            new_r.set(edgecolor="red")
            ax[i].add_patch(new_r)
        fig.suptitle(title_str)
        plt.savefig("step4" + imgname + ".jpeg")
        plt.show()
    return


def normalize_img_for_view(X, verbose=0):
    X = X.astype(np.float32)
    mn, mx = np.min(np.ravel(X)), np.max(np.ravel(X))
    if verbose > 1:
        print(f"1.min({mn}), max({mx})")
    X = X - mn
    mn, mx = np.min(np.ravel(X)), np.max(np.ravel(X))
    if verbose > 1:
        print(f"2.min({mn}), max({mx})")
    X = X / mx
    mn, mx = np.min(np.ravel(X)), np.max(np.ravel(X))
    if verbose > 1:
        print(f"3.min({mn}), max({mx})")
    X = X * 255
    mn, mx = np.min(np.ravel(X)), np.max(np.ravel(X))
    if verbose > 1:
        print(f"4.min({mn}), max({mx})")
    X = X.astype(np.uint8)
    mn, mx = np.min(np.ravel(X)), np.max(np.ravel(X))
    if verbose > 1:
        print(f"5.min({mn}), max({mx})")
    return X


def conv_im_w_box(img, img_box, verbose=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filter_count = 1
    kern_sz = np.shape(img_box)[0:2]

    if len(img_box.shape) == 3:
        filter_count = 3
    # conv = torch.nn.Conv2d(in_channels=filter_count, out_channels=1, kernel_size=kern_sz, stride=1, padding=(int(kern_sz[0]/2), int(kern_sz[1]/2)))
    conv = torch.nn.Conv2d(in_channels=filter_count, out_channels=1, kernel_size=kern_sz, stride=1, padding=0)

    if verbose > 1:
        print(f"device = {device}")
        print(f"img = {img.shape}")
        print(f"img_box = {img_box.shape}")
        print(f"kern_sz = {kern_sz}")
        print(f"conv.weight.shape = {conv.weight.shape}")
        print(f"filter_count = {filter_count}")

    img = torch.tensor(img)
    img_box = torch.tensor(img_box)
    if filter_count == 3:
        img = img.permute(2, 0, 1).unsqueeze(0)
        img_box.permute(2, 0, 1)
        for i in range(filter_count):
            conv.weight[0, i, :, :] = img_box[:, :, i]
    else:
        img = img.unsqueeze(0).unsqueeze(0)
        conv.weight[0, 0, :, :] = img_box

    conv.to(device)
    img = img.float().to(device)
    if verbose > 1:
        print(f"tensor img = {img.shape}")
    y1 = conv(img)
    if verbose > 1:
        print(f"y1 = {y1.shape}")
    y1 = y1.view(y1.shape[2], y1.shape[3], y1.shape[1])
    if verbose > 1:
        print(f"y1c = {y1.shape}")
    res_find = y1.cpu().detach().numpy().squeeze()
    return res_find


def crop_and_save(img_path, rect_crop):
    img = mpimg.imread(img_path)
    img_box = crop_rect_from_img(img, rect_crop)
    fname = img_path.replace(".jpg",
                             "") + f'_cropped_x{rect_crop.xy[0]}_y{rect_crop.xy[1]}_w{rect_crop.get_width()}_h{rect_crop.get_height()}.jpg'
    plt.imsave(fname, img_box)
    return


def crop_and_save_2(img, rect_crop):
    img_box = crop_rect_from_img(img, rect_crop)
    plt.imsave("im_base.jpg", img)
    fname = f'im_base_cropped_x{rect_crop.xy[0]}_y{rect_crop.xy[1]}_w{rect_crop.get_width()}_h{rect_crop.get_height()}.jpg'
    plt.imsave(fname, img_box)
    return


def locate_crop(img, crop, enforce_cpu=False, perc=None, max_w=None, verbose=0):
    img, perc = resize_img(img.astype(np.float32), perc=perc, max_w=max_w)
    crop, _ = resize_img(crop.astype(np.float32), perc=perc)
    if len(img.shape) == 3:
        img, crop = torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(crop, dtype=torch.float32)
        img, crop = img.permute(2, 1, 0).unsqueeze(0), crop.permute(2, 1, 0).unsqueeze(0)
    else:
        img, crop = torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(crop, dtype=torch.float32)
        img, crop = img.unsqueeze(0).unsqueeze(0), crop.unsqueeze(0).unsqueeze(0)

    img_sq, crop_sq = img * img, crop * crop
    crop_norm = crop / torch.sqrt(crop_sq.sum())

    sum_filter = torch.ones_like(crop_norm)

    device = 'cpu'
    if not enforce_cpu:
        device = check_and_return_device()

    if device == "cpu":
        img_crop_activation = torch.nn.functional.conv2d(img, crop_norm, stride=1, padding=0)
        img_sq_sum = torch.nn.functional.conv2d(img_sq, sum_filter, stride=1, padding=0)
    else:
        img_crop_activation = torch.nn.functional.conv2d(img.to(device), crop_norm.to(device), stride=1, padding=0)
        img_sq_sum = torch.nn.functional.conv2d(img_sq.to(device), sum_filter.to(device), stride=1, padding=0)

    cosine_distances = img_crop_activation / torch.sqrt(img_sq_sum)
    if device == "cpu":
        cosine_distances = cosine_distances.numpy().squeeze().squeeze()
    else:
        cosine_distances = cosine_distances.cpu().detach().numpy().squeeze().squeeze()

    coordinate = np.unravel_index(np.nanargmax(cosine_distances), cosine_distances.shape)

    foundxy = coordinate[0], coordinate[1]
    remapped_xy = int(round(coordinate[0] / perc)), int(round(coordinate[1] / perc))
    if verbose > 0:
        print(f"foundxy = {foundxy}")
        print(f"remapped_xy = {remapped_xy}")
        print(f"coordinate = {coordinate}")
    return {'col_w': remapped_xy[0], 'row_h': remapped_xy[1]}


def check_and_return_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    return device


def locate_rotated_crop_cosine(img, crop, deg_stride=5, enforce_cpu=False, verbose=0, perc=None, max_w=None):

    img, perc = resize_img(img.astype(np.float32), perc=perc, max_w=max_w)
    crop, _ = resize_img(crop.astype(np.float32), perc=perc)
    #new_h, new_w = crop.shape[0], crop.shape[1]

    deg_len = int(360 / deg_stride)
    if verbose>1:
        t_1s = datetime.now()
    if len(img.shape) == 3:
        a = torch.as_tensor(img, dtype=torch.float32).permute(2, 1, 0).unsqueeze(0)
        for i in range(1, deg_len):
            a = torch.cat((a, torch.as_tensor(rot_img_cv(img, i * deg_stride), dtype=torch.float32).permute(2, 1,
                                                                                                            0).unsqueeze(
                0)), 0)
        crop = torch.as_tensor(crop, dtype=torch.float32).permute(2, 1, 0).unsqueeze(0)
        if verbose > 1:
            t_2a = datetime.now()
            print(f'created image batch of {deg_len} images in {(t_2a - t_1s).total_seconds()} seconds')
    else:
        assert False, "not implemented"
    #    img, crop = torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(crop, dtype=torch.float32)
    #    img, crop = img.unsqueeze(0).unsqueeze(0), crop.unsqueeze(0).unsqueeze(0)

    a_sq, crop_sq = a * a, crop * crop
    crop_norm = crop / torch.sqrt(crop_sq.sum())

    sum_filter = torch.ones_like(crop_norm)

    device = 'cpu'
    if not enforce_cpu:
        device = check_and_return_device()

    if device == "cpu":
        img_crop_activation = torch.nn.functional.conv2d(a, crop_norm, stride=1, padding=0)
        img_sq_sum = torch.nn.functional.conv2d(a_sq, sum_filter, stride=1, padding=0)
    else:
        img_crop_activation = torch.nn.functional.conv2d(a.to(device), crop_norm.to(device), stride=1, padding=0)
        img_sq_sum = torch.nn.functional.conv2d(a_sq.to(device), sum_filter.to(device), stride=1, padding=0)

    if verbose > 1:
        t_3c = datetime.now()
        print(f'convolution completed in {(t_3c - t_2a).total_seconds()} seconds')

    cosine_distances = img_crop_activation / torch.sqrt(img_sq_sum)
    if verbose > 1:
        t_4d = datetime.now()
    if device == "cpu":
        cosine_distances = cosine_distances.numpy().squeeze()
    else:
        cosine_distances = cosine_distances.cpu().detach().numpy().squeeze()

    if verbose > 1:
        print(f'cosine_distances calced in {(t_4d - t_3c).total_seconds()} seconds')
        t_5u = datetime.now()

    coordinate = np.unravel_index(np.nanargmax(cosine_distances), cosine_distances.shape)
    if verbose > 1:
        print(f'unraveled in {(t_5u - t_4d).total_seconds()} seconds')
        t_6c = datetime.now()
        print(f'found best match in {(t_6c - t_5u).total_seconds()} seconds')

    foundxy = coordinate[1], coordinate[2]
    rotDegInd, rotDeg = coordinate[0], coordinate[0] * deg_stride
    remapped_xy = int(round(coordinate[1] / perc)), int(round(coordinate[2] / perc))
    if verbose > 0:
        print(f"foundxy = {foundxy}")
        print(f"remapped_xy = {remapped_xy}")
        print(f"coordinate = {coordinate}")
    if verbose > 0:
        #print out the surrounding of
        # img_crop_activation and cosine_distances
        try:
            print(f"img_crop_activation.shape {img_crop_activation.shape}")
            print(f"cosine_distances.shape {cosine_distances.shape}")
            colWFr, colWTo = coordinate[1]-5, coordinate[1]+5
            rowHFr, rowHTo = coordinate[2]-5, coordinate[2]+5
            print(f"rotDegInd({rotDegInd}), rotDeg({rotDeg}), colWFr({colWFr}), colWTo({colWTo}), rowHFr({rowHFr}), rowHTo({rowHTo})")
            cosDistBlock = pd.DataFrame(cosine_distances[rotDegInd, colWFr:colWTo, rowHFr:rowHTo])
            #print(f"img_crop_activation_partial\n{img_crop_activation[rotDegInd, 0, colWFr:colWTo, rowHFr:rowHTo]}")
            print(f"cosine_distances_partial\n{cosDistBlock}")
        except:
            print("*-"*15)
    coordinate = (coordinate[0], remapped_xy[0], remapped_xy[1])
    coord_all = {'rot_degree': coordinate[0] * deg_stride, 'col_w': coordinate[1], 'row_h': coordinate[2]}
    print(f"coordinate= {coordinate}")
    print(f"coord_all= {coord_all}")
    return coord_all, cosine_distances[rotDegInd, :, :].squeeze()


def search_iminim_simulate(img_path, rect_crop, verbose=1):
    ret_dict = {"rect_crop": copy(rect_crop)}
    crop_and_save(img_path, rect_crop)
    #  1. read the image
    img = mpimg.imread(img_path)
    ret_dict["img"] = img.copy()
    #  2. crop the rectangle from the image
    img_box = crop_rect_from_img(img, rect_crop)
    ret_dict["img_box"] = img_box.copy()
    ## NOW THE IMAGE AND IMG BOX ARE READY##
    # coordinate_1 = locate_crop(img, img_box)

    #  3. resize the image to max width allowed
    img, perc = resize_img(img.astype(np.float32), perc=None, max_w=1024)
    img_box, _ = resize_img(img_box.astype(np.float32), perc=perc)
    ret_dict["rs_img"] = img.copy()
    ret_dict["rs_img_box"] = img_box.copy()

    coordinate = locate_crop(img, img_box)

    #  4. get a -1/+1 copy of the original image and crop the part to search
    new_xy = round(rect_crop.xy[0] * perc), round(rect_crop.xy[1] * perc)
    new_h, new_w = round(rect_crop.get_height() * perc), round(rect_crop.get_width() * perc)
    if verbose > 0:
        print(f"new_xy = {new_xy}, new_h({new_h}, new_w({new_w})")
    rect_crop.set(xy=new_xy, width=new_w, height=new_h)
    ret_dict["rect_crop_new"] = copy(rect_crop)

    # crop_and_save_2(img, ret_dict["rect_crop_new"])
    img = img.astype(np.float32)
    img_box = img_box.astype(np.float32)
    ret_dict["s_rs_img"] = img.copy()
    ret_dict["s_rs_img_box"] = img_box.copy()

    modif_xy = round(rect_crop.xy[0]), round(rect_crop.xy[1])
    ret_dict["rect_crop_modif"] = copy(ret_dict["rect_crop_new"])
    ret_dict["rect_crop_modif"].set(xy=(int(modif_xy[0]), int(modif_xy[1])), width=new_w, height=new_h)
    if verbose > 0:
        print(f"rect_crop = {ret_dict['rect_crop']}")
        print(f"rect_crop_new = {ret_dict['rect_crop_new']}")
        print(f"modif_xy = {modif_xy}, new_h({new_h}, new_w({new_w})")
        print(f"rect_crop_modif = {ret_dict['rect_crop_modif']}")

    crop_norm = img_box / np.sqrt((img_box * img_box).sum())
    filt = np.ones(img_box.shape, dtype=np.float32)

    res_find_1 = conv_im_w_box(img, crop_norm, verbose=verbose)
    res_find_2 = conv_im_w_box(img * img, filt, verbose=verbose)

    ret_dict["res_find_1"] = normalize_img_for_view(res_find_1, verbose=verbose)
    ret_dict["res_find_2"] = normalize_img_for_view(res_find_2, verbose=verbose)

    res_find = res_find_1 / np.sqrt(res_find_2)
    ret_dict["res_find_f"] = normalize_img_for_view(res_find, verbose=verbose)

    mn, mx = np.min(np.ravel(res_find)), np.max(np.ravel(res_find))
    if verbose > 0:
        print(f"result min({mn}), max({mx})")
        print(f"res_find_shape = {res_find.shape}")
    rh, cw = np.unravel_index(res_find.argmax(), res_find.shape)
    foundxy = cw, rh
    remapped_xy = int(round(cw / perc)), int(round(rh / perc))
    if verbose > 0:
        print(f"foundxy = {foundxy}")
        print(f"remapped_xy = {remapped_xy}")
        print(f"coordinate = {coordinate}")
        print(f"given_xy = {ret_dict['rect_crop'].xy}")

    img_to_view = normalize_img_for_view(res_find, verbose=verbose)
    block_size = 20
    found_rect = get_rectangle(block_corner=(cw - block_size / 2, rh - block_size / 2),
                               block_wh={"w": block_size, "h": block_size})
    ret_dict["result_img"] = img_to_view.copy()
    ret_dict["result_rect"] = copy(found_rect)

    return ret_dict, res_find

    # print(f"will find in res_find({res_find.shape})")
    # rh, cw = np.unravel_index(res_find.argmax(), res_find.shape)
    # print(f"expected max value is {img_box.size}\n",
    #       f"faulty max can be {np.sum(img_box_bw.ravel())}\n",
    #       f"max val is {res_find[rh,cw]}")
    # block_corner = {"row_h": rh, "col_w": cw}
    # block_wh = {"w": img_box.shape[1], "h": img_box.shape[0]}
    # r = get_rectangle((block_corner["col_w"], block_corner["row_h"]), block_wh)
    # print(f"rect approximately at xcw({cw/perc}), yrh({rh/perc})")
    # return # r, img, img_box, res_find


def display_rotated_search(ret_dict, step, title_str="", imgname=""):
    plt.clf()
    if step == 0:
        fig, ax = plt.subplots(1, 3, figsize=(18, 9))
        ax[0].imshow(ret_dict["img"])
        new_r = copy(ret_dict["rect_crop"])
        new_r.set(edgecolor="r")
        ax[0].add_patch(new_r)
        ax[1].imshow(ret_dict["img_box"])
        ax[0].set_title(f"image input with size {ret_dict['img'].shape}")
        ax[1].set_title(f"r({new_r})\nbox size {ret_dict['img_box'].shape}")
        ax[2].imshow(ret_dict["img_rot"])
        ax[2].set_title(f"image rotated input with size {ret_dict['img_rot'].shape}")
        fig.suptitle(title_str)
        plt.savefig("Rstep0" + imgname + ".jpeg")
        plt.show()
    return


def locate_rotated_crop(img_path, rect_crop, rotate_deg, verbose=1):
    ret_dict = {"rect_crop": copy(rect_crop)}
    crop_and_save(img_path, rect_crop)
    #  1. read the image
    img = mpimg.imread(img_path)
    ret_dict["img"] = img.copy()
    #  2. crop the rectangle from the image
    img_box = crop_rect_from_img(img, rect_crop)
    ret_dict["img_box"] = img_box.copy()
    img_rot = rotate_img(img, rotate_deg)
    ret_dict["img_rot"] = img_rot.copy()

    return ret_dict


def find_center_of_box_in_rotated_image(rot_deg, img_shape, box_def: dict = None, rect_def: Rectangle = None,
                                        verbose=0):
    rm = create_rot_matrix(rot_deg)
    if box_def is not None and type(box_def) is dict and "box_corner" in box_def:
        boxdim = np.asarray([
            box_def["box_wh"]["w"] / 2,
            box_def["box_wh"]["h"] / 2,
            0.0], dtype=float)
        center_box = np.asarray([
            box_def["box_corner"]["col_w"] + boxdim[0],
            box_def["box_corner"]["row_h"] + boxdim[1],
            1.0], dtype=float)
        bc_wh_rot = {"w": int(box_def["box_wh"]["w"]),
                     "h": int(box_def["box_wh"]["h"])}
    elif rect_def is not None and type(rect_def) is plt.Rectangle:
        boxdim = np.asarray([
            rect_def.get_width() / 2,
            rect_def.get_height() / 2,
            0.0], dtype=float)
        center_box = np.asarray([
            rect_def.xy[0] + boxdim[0],
            rect_def.xy[1] + boxdim[1],
            1.0], dtype=float)
        bc_wh_rot = {"w": int(rect_def.get_width()),
                     "h": int(rect_def.get_height())}
    else:
        assert False, "either box_def or rect_def is necessary"

    center_img = np.asarray([
        img_shape[1] / 2,
        img_shape[0] / 2,
        0.0], dtype=float)

    if verbose > 0:
        print(f"center_box = {center_box}")
    center_box = center_box - center_img
    if verbose > 0:
        print(f"center_box relative to image center = {center_box}")
    rot_box = rotate_pixels([center_box], rm)
    if verbose > 0:
        print(f"rot_box - ci = {rot_box}")
    rot_box = rot_box.squeeze() + center_img - boxdim
    if verbose > 0:
        print(f"rot_box = {rot_box}")
        if rect_def is not None:
            print(f"original rectangle = {rect_def}")
    bc_rot = {"col_w": int(rot_box[0]),
              "row_h": int(rot_box[1])}

    r_rot = get_rectangle((int(bc_rot["col_w"]), int(bc_rot["row_h"])), bc_wh_rot)
    if verbose > 0:
        print(f"bc_rot = {bc_rot}")
        print(f"bc_wh_rot = {bc_wh_rot}")
        print(f"rotated rectangle = {r_rot}")
    return r_rot, bc_rot, bc_wh_rot


class img_rotator():
    def fit(self, image, assign_method='pick', nan_action='clip', rot_c=None, rot_deg=None):
        '''
        image  : the image that will be rotated
        assign_method : after the source pixel values found as float indices
                -'pick' : will basically pick rounded pixel indice
                -'interpolate': will interpolate the 4 source pixels
        nan_action: after rotation what will happen if the source pixel doesnt exist
                 -'clip' clip it to 0/w/h
                 -'remove' remove if any source value is out of reach
                 -'available' use only the available pixels
        '''
        self.image = image.astype('float')
        self.assign_method = assign_method
        self.nan_action = nan_action
        self.im_w, self.im_h = image.shape[0], image.shape[1]
        self.rot_c = [int(self.im_w / 2), int(self.im_h / 2)] if rot_c is None else rot_c
        self.rot_deg = 0 if rot_deg is None else rot_deg
        self.im_channel = 1 if len(image.shape) < 3 else image.shape[2]

    def construct_weights(self, rot_c=None, rot_deg=None):
        self.rot_c = self.rot_c if rot_c is None else rot_c
        self.rot_deg = self.rot_deg if rot_deg is None else rot_deg
        self.get_rotated_pixel_vals()

    @staticmethod
    def _get_pixels_of_image_given_center(w, h, c):
        xs = np.tile(np.array([np.arange(w)]), (h, 1))
        ys = np.tile(np.array([np.arange(h)]), (w, 1)).transpose()
        pixel_pts = convert_points_3col(np.c_[ys.ravel(), xs.ravel()])
        if len(c) == 2:
            c.append(0)
        center_add = c
        # print(f"pixel_pts.shape = {pixel_pts.shape}")
        # print(f"c.shape = {np.shape(c)}, c({c}")
        return pixel_pts - c, center_add

    @staticmethod
    def _get_rotated_pixel_vals(im_w, im_h, rot_c, rot_deg):
        rm = create_rot_matrix(rot_deg)
        pix_vals_original, center_add = img_rotator._get_pixels_of_image_given_center(w=im_w, h=im_h, c=rot_c)
        rot_pixels = rotate_pixels(pix_vals_original, rm)
        return rot_pixels, pix_vals_original, center_add

    def get_rotated_pixel_vals(self):
        self.rot_pixels, self.pix_vals_original, self.center_add = img_rotator._get_rotated_pixel_vals(im_w=self.im_w,
                                                                                                       im_h=self.im_h,
                                                                                                       rot_c=self.rot_c,
                                                                                                       rot_deg=self.rot_deg)
        return self.rot_pixels, self.pix_vals_original, self.center_add

    def arrange_borders(self, x, y, trsh=0.00001):
        x[x >= self.im_w - 1] = x[x >= self.im_w - 1] - trsh
        y[y >= self.im_h - 1] = y[y >= self.im_h - 1] - trsh
        xSlct = ((x <= trsh).astype(int) * (x >= -trsh).astype(int)).astype(bool)
        ySlct = ((y <= trsh).astype(int) * (y >= -trsh).astype(int)).astype(bool)
        x[xSlct] = x[xSlct] + trsh
        y[ySlct] = y[ySlct] + trsh
        return x, y

    def expand_4_channels(self, imSourceIdx, imTargetIdx):
        # print(f"self.im_channel = {self.im_channel}")
        if self.im_channel == 1:
            return imSourceIdx, imTargetIdx
        # print(f"BEFORE \n imSourceIdx\n{imSourceIdx}\nimTargetIdx\n{imTargetIdx}")
        print("\n*******\n", (self.im_w * self.im_h * np.arange(self.im_channel) + np.repeat(imTargetIdx,
                                                                                             self.im_channel).reshape(
            -1, self.im_channel)).T.ravel())
        imTargetIdx = np.arange(0, self.im_w * self.im_h * self.im_channel)
        print(f"{imTargetIdx}*****\n")

        map2d_c = imTargetIdx.reshape(self.im_h * self.im_w, self.im_channel)
        # print(f"map2d_c = {map2d_c.shape}")
        print(f"map2d_c = {map2d_c.shape}\n{map2d_c}")
        map2d_c = map2d_c[imSourceIdx, :]
        print(f"map2d_c = {map2d_c.shape}")
        # print(f"map2d_c = {map2d_c.shape}\n{map2d_c}")
        imSourceIdx = map2d_c.ravel()
        # print(f"AFTER \n imSourceIdx\n{imSourceIdx}\nimTargetIdx\n{imTargetIdx}")
        return imSourceIdx, imTargetIdx

    def apply_rotation(self):
        self.construct_weights()
        xyz = np.asarray(self.center_add, dtype=float) + self.rot_pixels.astype(float)
        x, y = xyz[:, 0], xyz[:, 1]
        xf, yf = None, None
        xw, yw = None, None
        if self.assign_method == 'pick':
            x, y = np.round(x, 0).astype(int), np.round(y, 0).astype(int)
            if self.nan_action == 'clip':
                x = np.clip(x, 0, self.im_w - 1).astype('int')
                y = np.clip(y, 0, self.im_h - 1).astype('int')
                imSourceIdx = np.ravel_multi_index([x, y], (self.im_w, self.im_h))
                imTargetIdx = np.arange(0, len(imSourceIdx), dtype=int)
            elif self.nan_action in ['remove', 'available']:
                iSoTa = [[s, int(np.ravel_multi_index([[i], [j]], (self.im_w, self.im_h)))] for s, (i, j) in
                         enumerate(zip(x, y)) if (i < self.im_w and i >= 0 and j < self.im_h and j >= 0)]
                iSoTa = np.asarray(iSoTa)
                imTargetIdx = iSoTa[:, 0]
                imSourceIdx = iSoTa[:, 1]
            self.imRot = np.zeros_like(self.image.ravel())
            if self.im_channel == 1:
                self.imRot[imTargetIdx] = self.image.ravel()[imSourceIdx]
            else:
                imSourceIdx, imTargetIdx = self.expand_4_channels(imSourceIdx, imTargetIdx)
                self.imRot[imTargetIdx] = self.image.ravel()[imSourceIdx]
                # self.imRot[imTargetIdx] = self.image.transpose(2,1,0).ravel()[imSourceIdx].transpose(0,1,2)
            self.imRot = self.imRot.reshape(self.image.shape)
        elif self.assign_method == 'interpolate':
            if self.nan_action == 'clip':
                xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
                xw, yw = x - xf, y - yf
                # print(f"im shape:\n{self.image.shape}")
                imRot = [self.image[np.clip(xf + i % 2, 0, self.im_w - 1).astype('int'),
                                    np.clip(yf + int(i / 2), 0, self.im_h - 1).astype('int'), rgb] * (
                                 (1.0 - i % 2) + ((i % 2 - .5) * 2) * xw) * (
                                 (1 - int(i / 2)) + ((int(i / 2) - .5) * 2) * yw) for rgb in range(self.im_channel) for
                         i in range(4)]
                # print(f"imRot:\n{np.shape(imRot)}\n{imRot}")
                imRot = np.reshape(imRot, (self.im_channel, 4, len(xf))).transpose((1, 2, 0))
                # print(f"imRot:\n{np.shape(imRot)}\n{imRot}")
                imRot = np.sum(imRot, axis=0)
                # print(f"imRot:\n{np.shape(imRot)}\n{imRot}")
                self.imRot = imRot.reshape(self.image.shape)
            elif self.nan_action == 'remove':
                x, y = self.arrange_borders(x, y, trsh=1e-8)
                xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
                xw, yw = x - xf, y - yf
                slct = ((xf >= 0).astype(int) * (xf < self.im_w - 1).astype(int) * (yf >= 0).astype(int) * (
                        yf < self.im_h - 1).astype(int)).astype(bool)
                print(f"select=\n{slct}")
                imRot = [self.image[(xf + i % 2)[slct], (yf + int(i / 2))[slct], rgb] * (
                        (1.0 - i % 2) + ((i % 2 - .5) * 2) * xw[slct]) * (
                                 (1 - int(i / 2)) + ((int(i / 2) - .5) * 2) * yw[slct]) for rgb in
                         range(self.im_channel) for i in range(4)]
                print(f"imRot1:\n{np.shape(imRot)}\n{imRot}")
                imRot = np.reshape(imRot, (self.im_channel, 4, sum(slct.astype(int)))).transpose((1, 2, 0))
                print(f"imRot2:\n{np.shape(imRot)}\n{imRot}")
                imRot = np.sum(imRot, axis=0)
                print(f"imRot3:\n{np.shape(imRot)}\n{imRot}")
                slct2 = np.repeat(slct, self.im_channel).reshape(-1, self.im_channel)
                print(f"slct2=\n{slct2}")
                self.imRot = np.zeros_like(self.image.ravel(), dtype=float)
                self.imRot[slct2.ravel()] = imRot.ravel()
                self.imRot = self.imRot.reshape(self.image.shape)
            elif self.nan_action == 'available':
                assert False, "not implemented"
        return {"xf": xf, "yf": yf, "xw": xw, "yw": yw, "x": x, "y": y}
