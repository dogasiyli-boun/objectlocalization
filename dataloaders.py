import wget
import os
import tarfile
import gzip
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import rotnetfuncs as rnf

class GenericImageDataset(Dataset):
    def __init__(self, images, labels, datasetname, istrain=True, transform=None):
        self.transform = transform
        self.datasetname = datasetname
        ids = np.arange(0, len(labels))
        self.images = images
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx, :] # Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        ids = self.ids[idx]
        sample = {'image': image, 'label': label, 'id': ids}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def untar_file_into_folder(dataFolderCurrent, filename, extractInto):
    extrFile = os.path.join(dataFolderCurrent, filename)
    tf = tarfile.open(extrFile)
    extrIntoFold = os.path.join(dataFolderCurrent, extractInto)
    if not os.path.isdir(extrIntoFold):
        os.mkdirs(extrIntoFold)
    try:
        tf.extractall(extrIntoFold)
    except:
        pass
    tf.close()
    return

def download_data(name, rootFold=""):
    dataFolderCurrent = os.path.join(rootFold, "dataset", name)
    if not os.path.isdir(dataFolderCurrent):
        os.mkdir(dataFolderCurrent)
    if name=="fashion-mnist":
        fnames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        fidents = ["trainImages", "trainLabels", "testImages", "testLabels"]
        for fname, fident in zip(fnames, fidents):
            if not os.path.isfile(os.path.join(dataFolderCurrent, fname)):
                wget.download(url="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"+fname, out=dataFolderCurrent)
    elif name=="mnist":
        fnames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        for fname in fnames:
            if not os.path.isfile(os.path.join(dataFolderCurrent, fname)):
                wget.download(url="http://yann.lecun.com/exdb/mnist/"+fname, out=dataFolderCurrent)

def read_mnist_data(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def rotate_Xmnist_data(images, labels, deg_inc=1, crop_center=False, crop_largest_rect=False):
    n = len(labels)
    for i in range(n):
        im = images[i, :].reshape(28, 28)
        rotation_angle = np.random.randint(360/deg_inc)
        rotated_image = rnf.generate_rotated_image(
            im,
            rotation_angle*deg_inc,
            size=(28, 28),
            crop_center=crop_center,
            crop_largest_rect=crop_largest_rect
        )
        images[i, :] = rotated_image.flatten()
        labels[i] = rotation_angle
    return images, labels

def load_data(name, rootFold="", train=True, rot_deg_inc=0):
    if name in  ["fashion-mnist", "mnist"]:
        download_data(name, rootFold=rootFold)
        dataFolderCurrent = os.path.join(rootFold, "dataset", name)
        fnames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        if train:
            images, labels = read_mnist_data(images_path=os.path.join(dataFolderCurrent, fnames[0]), labels_path=os.path.join(dataFolderCurrent, fnames[1]))
        else:
            images, labels = read_mnist_data(images_path=os.path.join(dataFolderCurrent, fnames[2]), labels_path=os.path.join(dataFolderCurrent, fnames[3]))
        if rot_deg_inc > 0:
            images = images.copy()
            labels = labels.copy()
            images.setflags(write=1)
            labels.setflags(write=1)
            images, labels = rotate_Xmnist_data(images, labels, deg_inc=rot_deg_inc, crop_center=False, crop_largest_rect=False)
            images.setflags(write=0)
            labels.setflags(write=0)
    return images, labels

def load_datasets(name, rootFold="", rot_deg_inc=0):
    if name in  ["fashion-mnist", "mnist"]:
        dataFolderCurrent = os.path.join(rootFold, "dataset", name)
        images_tr, labels_tr = load_data(name, rootFold=rootFold, train=True, rot_deg_inc=rot_deg_inc)
        images_te, labels_te = load_data(name, rootFold=rootFold, train=False, rot_deg_inc=rot_deg_inc)
        data = {}
        data["tr"] = GenericImageDataset(images_tr, labels_tr, name + "_tr", True, None)
        data["te"] = GenericImageDataset(images_te, labels_te, name + "_tr", True, None)
    return data

def mnist_show_examples(train_data):
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        sample = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(sample["label"])
        plt.axis("off")
        plt.imshow(sample["image"].reshape(28,28), cmap="gray")
    plt.show()
    plt.close()