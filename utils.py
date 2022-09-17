import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from copy import copy
from matplotlib.patches import Rectangle
from scipy import ndimage
import math
from torchvision.transforms import Resize as tResize, ToPILImage as tToPILImage, ToTensor as tToTensor, Compose as tCompose

def mat_mul(A, B):
    # cast into <a href="https://geekflare.com/numpy-reshape-arrays-in-python/">NumPy array</a> using np.array()
    return np.array([[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A])

def create_rot_matrix(deg):
    rad = math.radians(deg)
    print(f"deg({deg}) = rad({rad})")
    return [[np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]]

def convert_points_3col(pt_list):
    one_list = np.ones((np.shape(pt_list)[0],1))
    #print(f"one_list shape = {np.shape(one_list)}")
    #print(f"pt_list shape = {np.shape(pt_list)}")
    return np.c_[np.array(pt_list), one_list]

def plot_circle_given_center_point():
    return

def get_pixels_of_image_given_center(w, h, c):
    xs = np.tile(np.array([np.arange(w)]), (h,1))
    ys = np.tile(np.array([np.arange(h)]), (w,1)).transpose()
    pixel_pts = convert_points_3col(np.c_[ys.ravel(), xs.ravel()])
    c.append(0)
    center_add = c
    return pixel_pts - c, center_add

def get_indices_of_image(w, h):
    # from itertools import product
    # def get_indices_of_image(w, h):
    #    xs = range(w)
    #    ys = range(h)
    #    return np.array(list(product(xs, ys)))
    xs = np.tile(np.array([np.arange(w)]), (h,1))
    ys = np.tile(np.array([np.arange(h)]), (w,1)).transpose()
    return np.c_[ys.ravel(), xs.ravel()]

def rotate_pixels(idx, rm):
    return mat_mul(idx, rm)

def get_rotated_pixel_vals(im_w, im_h, rot_c, rot_deg):
    rm = create_rot_matrix(rot_deg)
    pix_vals_original, center_add = get_pixels_of_image_given_center(w=im_w, h=im_h, c=rot_c)
    rot_pixels = rotate_pixels(pix_vals_original, rm)
    return rot_pixels, pix_vals_original, center_add

def rotate_img(img, degree):
    rotated_img = ndimage.rotate(img, degree*60)
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
    o[x2[:, 0], x2[:, 1]] =  1.0
    return o

def crop_rect_from_img(img, r):
    box = {
        "col_w_beg": r.xy[0],
        "row_h_beg": r.xy[1],
        "col_w_end": r.xy[0]+r.get_width(),
        "row_h_end": r.xy[1]+r.get_height(),
        "col_width": r.get_width(),
        "row_height": r.get_height(),
    }
    #rows = np.arange(box["row_h_beg"], box["row_h_end"])
    #cols = np.arange(box["col_w_beg"], box["col_w_end"])
    #print("rows : ", rows)
    #print("cols : ", cols)
    if len(np.shape(img)) == 2:
        img_box = img[box["row_h_beg"]:box["row_h_end"], box["col_w_beg"]:box["col_w_end"]].copy()
    elif len(np.shape(img)) == 3:
        img_box = img[box["row_h_beg"]:box["row_h_end"], box["col_w_beg"]:box["col_w_end"], :].copy()
    return img_box

def crop_rect_for_bw_search(img, r):
    return crop_rect_from_img(make_uint8_bw(img), r)

def crop_and_show(img, r):
    img_box = crop_rect_from_img(img, r)
    plt.clf()
    fig, ax = plt.subplots(1, 2,  figsize=(20, 10), sharex=True)
    ax[0].imshow(img)
    new_r=copy(r)
    ax[0].add_patch(new_r)
    ax[1].imshow(img_box)
    plt.show()
    return img_box

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float32)

def resize_img(img, perc=None, max_w=None, max_h=None):
    h, w = img.shape[0], img.shape[1]
    if perc is None and max_w is not None and w > max_w:
        perc = max_w/w
    elif perc is None and max_h is not None and h > max_h:
        perc = max_w/w
    elif perc is None:
        perc = 1.0
    w_new = int(w*perc)
    h_new = int(h*perc)
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
    #conv.weight[1, 0, :, :] = torch.from_numpy(1 - img_box)

    conv.to(device)
    img_1 = torch.from_numpy(np.asarray(img_bw, dtype=np.float32)).to(device)
    #img_2 = torch.from_numpy(np.asarray(1 - img_bw, dtype=np.float32)).to(device)
    y1 = conv(img_1.view(1, 1, img_1.shape[0], img_1.shape[1]))
    #·y2 = conv(img_2.view(1, 1, img_2.shape[0], img_2.shape[1]))
    res_img_1 = y1.cpu().detach().numpy().squeeze()
    #res_img_2 = y2.cpu().detach().numpy().squeeze()
    res_find = res_img_1  # [0, :, :]  - res_img_1[1, :, :] + res_img_2[1, :, :] - res_img_2[0, :, :]
    print(f"will find in res_find({res_find.shape})")
    rh, cw = np.unravel_index(res_find.argmax(), res_find.shape)
    print(f"expected max value is {img_box.size}\n",
          f"faulty max can be {np.sum(img_box_bw.ravel())}\n",
          f"max val is {res_find[rh,cw]}")
    block_corner = {"row_h": rh, "col_w": cw}
    block_wh = {"w": img_box.shape[1], "h": img_box.shape[0]}
    r = get_rectangle((block_corner["col_w"], block_corner["row_h"]), block_wh)
    print(f"rect approximately at xcw({cw/perc}), yrh({rh/perc})")
    return r, img, img_box, res_find



def display_iminim_results(ret_dict, step, title_str="", imgname=""):
    plt.clf()
    if step == 0:
        fig, ax = plt.subplots(1, 2,  figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(ret_dict["img"])
        new_r = copy(ret_dict["rect_crop"])
        new_r.set(edgecolor="r")
        ax[0].add_patch(new_r)
        ax[1].imshow(ret_dict["img_box"])
        ax[0].set_title(f"image input with size {ret_dict['img'].shape}")
        ax[1].set_title(f"r({new_r})\nbox size {ret_dict['img_box'].shape}")
        fig.suptitle(title_str)
        plt.savefig("step0"+imgname+".jpeg")
        plt.show()
    if step == 1:
        fig, ax = plt.subplots(1, 2,  figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
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
        fig, ax = plt.subplots(1, 2,  figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
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
    X = X/mx
    mn, mx = np.min(np.ravel(X)), np.max(np.ravel(X))
    if verbose > 1:
        print(f"3.min({mn}), max({mx})")
    X = X*255
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

    if len(img_box.shape)==3:
        filter_count = 3
    conv = torch.nn.Conv2d(in_channels=filter_count, out_channels=1, kernel_size=kern_sz, stride=1, padding=(int(kern_sz[0]/2), int(kern_sz[1]/2)))

    if verbose > 1:
        print(f"device = {device}")
        print(f"img = {img.shape}")
        print(f"img_box = {img_box.shape}")
        print(f"kern_sz = {kern_sz}")
        print(f"conv.weight.shape = {conv.weight.shape}")
        print(f"filter_count = {filter_count}")

    if filter_count > 1:
        for i in range(filter_count):
            conv.weight[0, i, :, :] = torch.from_numpy(img_box[:, :, i])
    else:
        conv.weight[0, 0, :, :] = torch.from_numpy(img_box)
    conv.to(device)

    img = torch.tensor(img)
    if filter_count==3:
        img = img.permute(2, 0, 1).unsqueeze(0)
    else:
        img = img.unsqueeze(0).unsqueeze(0)
    img = img.float().to(device)

    if verbose > 1:
        print(f"tensor img = {img.shape}")
    y1 = conv(img)
    if verbose > 1:
        print(f"y1 = {y1.shape}")
    y1 = y1.view(y1.shape[2], y1.shape[3], y1.shape[1])
    if verbose > 1:
        print(f"y1 = {y1.shape}")
    res_find = y1.cpu().detach().numpy().squeeze()
    return res_find

def search_iminim_simulate(img_path, rect_crop, verbose=1):
    ret_dict = {"rect_crop": copy(rect_crop)}
    #  1. read the image
    img = mpimg.imread(img_path)
    ret_dict["img"] = img.copy()
    #  2. crop the rectangle from the image
    img_box = crop_rect_from_img(img, rect_crop)
    ret_dict["img_box"] = img_box.copy()
    ## NOW THE IMAGE AND IMG BOX ARE READY##

    #  3. resize the image to max width allowed
    img, perc = resize_img(img.astype(np.float32), perc=None, max_w=1024)
    img_box, _ = resize_img(img_box.astype(np.float32), perc=perc)
    ret_dict["rs_img"] = img.copy()
    ret_dict["rs_img_box"] = img_box.copy()

    #  4. get a -1/+1 copy of the original image and crop the part to search
    img = 0.5-img.astype(np.float32)/np.max(img)
    new_xy = round(rect_crop.xy[0]*perc), round(rect_crop.xy[1]*perc)
    new_h, new_w = round(rect_crop.get_height()*perc), round(rect_crop.get_width()*perc)
    if verbose > 0:
        print(f"new_xy = {new_xy}, new_h({new_h}, new_w({new_w})")
    rect_crop.set(xy=new_xy, width=new_w, height=new_h)
    ret_dict["rect_crop_new"] = copy(rect_crop)
    img_box = crop_rect_from_img(img, rect_crop)
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

    res_find_1 = conv_im_w_box(img, img_box, verbose=verbose)
    filt = np.ones(img_box.shape, dtype=np.float32)
    res_find_2 = conv_im_w_box(img*img, filt, verbose=verbose)

    ret_dict["res_find_1"] = normalize_img_for_view(res_find_1, verbose=verbose)
    ret_dict["res_find_2"] = normalize_img_for_view(res_find_2, verbose=verbose)

    res_find_1[res_find_1<0] = 0.0
    res_find_2[res_find_2<0] = 0.0
    res_find_1 = np.sqrt(res_find_1)
    res_find_2 = np.sqrt(res_find_2)
    res_find = res_find_1/res_find_2
    ret_dict["res_find_f"] = normalize_img_for_view(res_find, verbose=verbose)

    mn, mx = np.min(np.ravel(res_find)), np.max(np.ravel(res_find))
    if verbose > 0:
        print(f"result min({mn}), max({mx})")
        print(f"res_find_shape = {res_find.shape}")
    rh, cw = np.unravel_index(res_find.argmax(), res_find.shape)
    foundxy = cw, rh
    remapped_xy = int(round(cw/perc)), int(round(rh/perc))
    if verbose > 0:
        print(f"foundxy = {foundxy}")
        print(f"remapped_xy = {remapped_xy}")
        print(f"given_xy = {ret_dict['rect_crop'].xy}")

    img_to_view = normalize_img_for_view(res_find, verbose=verbose)
    block_size = 20
    found_rect = get_rectangle(block_corner=(cw-block_size/2, rh-block_size/2), block_wh={"w": block_size, "h": block_size})
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


