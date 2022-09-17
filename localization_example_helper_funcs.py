from os.path import isfile, isdir, join as path_join, splitext
from os import listdir, mkdir
from zipfile import ZipFile
from pandas import DataFrame as pd_df
from csv import reader as csv_reader

from xml.dom import minidom
import cv2
import matplotlib.pyplot as plt
import random
from numpy import clip as np_clip, array as np_array, expand_dims as np_expand_dims
from numpy.random import randint as np_randint

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import glob

from helperFuncs import get_num_correct, check_and_return_device, print_tensor_size

class Dataset( ):
    def __init__(self, train_images, train_labels, train_boxes):
        # torch.permute(torch.from_numpy(train_images),(0,3,1,2)).float()
        self.images = torch.from_numpy(train_images).permute((0, 3, 1, 2)).float()

        self.labels = torch.from_numpy(train_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(train_boxes).float()

    def __len__(self):
        return len(self.labels)

    # To return x,y values in each iteration over dataloader as batches.

    def __getitem__(self, idx):
        return (self.images[idx],
              self.labels[idx],
              self.boxes[idx])

# Inheriting from Dataset class
class ValDataset(Dataset):
    def __init__(self, val_images, val_labels, val_boxes):
        # torch.permute(torch.from_numpy(val_images),(0,3,1,2)).float()
        self.images = torch.from_numpy(val_images).permute((0, 3, 1, 2)).float()
        self.labels = torch.from_numpy(val_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(val_boxes).float()

class SampleNetwork(nn.Module):
    def __init__(self):
        super(SampleNetwork, self).__init__()

        # CNNs for rgb images
        # self.convs = [nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        #               nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
        #               nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5),
        #               nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5),
        #               nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)

        # Connecting CNN outputs with Fully Connected layers for classification
        self.class_fc1 = nn.Linear(in_features=1728, out_features=240)
        #self.class_out = nn.Linear(in_features=240, out_features=2)

        # Connecting CNN outputs with Fully Connected layers for bounding box
        #self.box_fc1 = nn.Linear(in_features=1728, out_features=240)
        #self.box_out = nn.Linear(in_features=240, out_features=4)
    def forward(self, t):
        print_tensor_size("*1.input tensor", t)

        # for j in range(5):
        #     t = self.convs[j](t)
        #     print_tensor_size("*{:d}.after conv tensor".format(j), t)
        #     t = F.relu(t)
        #     t = F.max_pool2d(t, kernel_size=2, stride=2)
        #     print_tensor_size("*{:d}.after maxpool tensor".format(j+1), t)
        j=0
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print_tensor_size("*{:d}.after maxpool tensor".format(j + 1), t)

        j+=1
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print_tensor_size("*{:d}.after maxpool tensor".format(j + 1), t)

        j+=1
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print_tensor_size("*{:d}.after maxpool tensor".format(j + 1), t)

        j+=1
        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print_tensor_size("*{:d}.after maxpool tensor".format(j + 1), t)

        j+=1
        t = self.conv5(t)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=4, stride=2)
        print_tensor_size("*{:d}.after avgpool tensor".format(j + 1), t)

        j+=1
        t = torch.flatten(t, start_dim=1)
        print_tensor_size("*{:d}.after flatten tensor".format(j+1), t)

        class_t = F.relu(self.class_fc1(t))
        #class_t = F.softmax(self.class_out(class_t), dim=1)

        print_tensor_size("*4.1.class_t", class_t)

        #box_t = F.relu(self.box_fc1(t))
        #box_t = F.sigmoid(self.box_out(box_t))
        #print_tensor_size("*4.2.box_t", box_t)

        return [class_t] #[class_t, box_t]

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # CNNs for rgb images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)

        # Connecting CNN outputs with Fully Connected layers for classification
        self.class_fc1 = nn.Linear(in_features=1728, out_features=240)
        self.class_fc2 = nn.Linear(in_features=240, out_features=120)
        #self.class_out = nn.Linear(in_features=120, out_features=2)

        # Connecting CNN outputs with Fully Connected layers for bounding box
        self.box_fc1 = nn.Linear(in_features=1728, out_features=240)
        self.box_fc2 = nn.Linear(in_features=240, out_features=120)
        self.box_out = nn.Linear(in_features=120, out_features=4)
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv5(t)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=4, stride=2)

        t = torch.flatten(t, start_dim=1)

        class_t = self.class_fc1(t)
        class_t = F.relu(class_t)

        class_t = self.class_fc2(class_t)
        class_t = F.relu(class_t)

        class_t = F.softmax(self.class_out(class_t), dim=1)

        box_t = self.box_fc1(t)
        box_t = F.relu(box_t)

        box_t = self.box_fc2(box_t)
        box_t = F.relu(box_t)

        box_t = self.box_out(box_t)
        box_t = F.sigmoid(box_t)

        return [class_t, box_t]

def preprocess(img, image_size=256):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0

    # Expand dimensions as predict expect image in batches
    image = np_expand_dims(image, axis=0)
    return image

def postprocess_results(results, num_to_labels={0: 'cat', 1: 'dog'}):
    # Split the results into class probabilities and box coordinates
    [class_probs, bounding_box] = results

    # First let's get the class label

    # The index of class with the highest confidence is our target class
    class_index = torch.argmax(class_probs).item()

    # Use this index to get the class name.
    print("class_index = ({}) with type {}".format(class_index, type(class_index)))
    class_label = num_to_labels[class_index]

    # Now you can extract the bounding box too.

    # Get the height and width of the actual image
    h, w = 256, 256

    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]

    # # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)

    # return the lable and coordinates
    return class_label, (x1, y1, x2, y2), torch.max(class_probs) * 100

def load_model(model_epoch_id, device):
    model = Network()
    model = model.to(device)
    model.load_state_dict(torch.load("models/model_ep{:d}.pth".format(model_epoch_id)))
    return model

# We will use this function to make prediction on images.
def predict(image_path, model_epoch_id=29, scale=0.5, fontwrite=cv2.FONT_HERSHEY_COMPLEX):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_epoch_id, device)
    model.eval()

    # Reading Image
    img = cv2.imread(image_path)

    # Before we can make a prediction we need to preprocess the image.
    processed_image = preprocess(img)

    # result = model(torch.permute(torch.from_numpy(processed_image).float(),(0,3,1,2)).to(device))
    result = model(torch.from_numpy(processed_image).permute((0, 3, 1, 2)).float().to(device))

    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence = postprocess_results(result)

    gt = image_path.split("/")[-1].split(".")[0]

    symb = "=" if gt==label else "x"
    col = (0, 255, 100) if gt==label else (255, 0, 100)
    col2 = (0, 255, 0) if gt==label else (0, 255, 0)

    # Now annotate the image
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    cv2.putText(img, 'pr({}){}({})gt'.format(label,symb,gt), (30, int(35 * scale)), fontwrite, scale, col2, 1)
    cv2.putText(img, 'conf({:4.2f})'.format(confidence), (30, int(65 * scale)), fontwrite, scale, col2, 1)

    return img

def test(model, valdataloader, optimizer, device):
    tot_loss = 0
    tot_correct = 0
    num_samples = 0
    model.eval()
    for batch, (x, y, z) in enumerate(valdataloader):
        # Converting data from cpu to GPU if available to improve speed
        x, y, z = x.to(device), y.to(device), z.to(device)
        # Sets the gradients of all optimized tensors to zero
        optimizer.zero_grad()
        with torch.no_grad():
            [y_pred, z_pred] = model(x)

            # Compute loss (here CrossEntropyLoss)
            class_loss = F.cross_entropy(y_pred, y)
            box_loss = F.mse_loss(z_pred, z)
            # Compute loss (here CrossEntropyLoss)

        tot_loss += (class_loss.item() + box_loss.item())
        tot_correct += get_num_correct(y_pred, y)
        num_samples += len(y)
        # print("Test batch:", batch + 1, " epoch: ", epoch, " ", (time.time() - train_start) / 60, end='\r')
    correct_percent = 100*(tot_correct/num_samples)

    return correct_percent, tot_loss

def train(dataloader, valdataloader, model=None, num_of_epochs = 30, start_from_scratch=False):
    device = check_and_return_device()
    max_epoch = 0
    if model is None and start_from_scratch:
        model = Network()
        model = model.to(device)
        print("Model to be trained is created from scratch")
    elif model is None and not start_from_scratch:
        epochs_saved_list = [int(str(f).replace("model_ep", "").replace(".pth", "")) for f in listdir("models/") if f.endswith(".pth")]
        max_epoch = max(epochs_saved_list)
        model = load_model(max_epoch, device)
        print("Model to be trained is loaded as the last model({:d}) in models folder".format(max_epoch))
    else:
        print("Model to be trained is passed as argument")

    # Defining the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = []
    losses = []
    acc_list = []
    # Creating a directory for storing models
    if not isdir("models"):
        mkdir('models')
    for epoch in range(num_of_epochs):
        train_start = time.time()
        model.train()
        for batch, (x, y, z) in enumerate(dataloader):
            # Converting data from cpu to GPU if available to improve speed
            x, y, z = x.to(device), y.to(device), z.to(device)
            # Sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            [y_pred, z_pred] = model(x)
            # Compute loss (here CrossEntropyLoss)
            class_loss = F.cross_entropy(y_pred, y)
            box_loss = F.mse_loss(z_pred, z)
            (box_loss + class_loss).backward()
            # class_loss.backward()
            optimizer.step()
            print("Train batch:", batch + 1, " epoch: ", max_epoch+epoch, " ",
                  (time.time() - train_start) / 60, end='\r')

        correct_percent, tot_loss = test(model, valdataloader, optimizer, device)

        epochs.append(max_epoch+epoch)
        losses.append(tot_loss)
        acc_list.append(correct_percent)
        str_2p = "Epoch {:3d}:Accuracy {:5.3f}/ loss:{:5.3f}/ time:{:5.3f} mins".format(max_epoch+epoch, correct_percent, tot_loss, (time.time() - train_start) / 60)
        print(str_2p)
        torch.save(model.state_dict(), path_join("models", "model_ep" + str(max_epoch+epoch + 1) + ".pth"))
    return model, acc_list

def check_localization_dataset():
    if (isfile("localization_dataset.zip")) and not isdir("dataset"):
        with ZipFile("localization_dataset.zip", "r") as zip_ref:
            zip_ref.extractall()
    elif isdir("dataset") and isdir("dataset/annot") and isdir("dataset/images"):
        print("Data is ready")
    else:
        raise SystemExit("Data needs to be checked")

    return True

def extract_xml_contents(annot_directory, image_dir):
    file = minidom.parse(annot_directory)

    # Get the height and width for our image
    height, width = cv2.imread(image_dir).shape[:2]

    # Get the bounding box co-ordinates
    xmin = file.getElementsByTagName('xmin')
    x1 = float(xmin[0].firstChild.data)

    ymin = file.getElementsByTagName('ymin')
    y1 = float(ymin[0].firstChild.data)

    xmax = file.getElementsByTagName('xmax')
    x2 = float(xmax[0].firstChild.data)

    ymax = file.getElementsByTagName('ymax')
    y2 = float(ymax[0].firstChild.data)

    class_name = file.getElementsByTagName('name')

    if class_name[0].firstChild.data == "cat":
        class_num = 0
    else:
        class_num = 1

    files = file.getElementsByTagName('filename')
    filename = files[0].firstChild.data

    # Return the extracted attributes
    return filename, width, height, class_num, x1, y1, x2, y2

# Function to convert XML files to CSV
def xml_to_csv():
  # List containing all our attributes regarding each image
  xml_list = []

  # We loop our each class and its labels one by one to preprocess and augment
  image_dir = 'dataset/images'
  annot_dir = 'dataset/annot'

  # Get each file in the image and annotation directory
  mat_files = listdir(annot_dir)
  img_files = listdir(image_dir)

  # Loop over each of the image and its label
  for mat, image_file in zip(mat_files, img_files):
      # Full mat path
      mat_path = path_join(annot_dir, mat)
      # Full path Image
      img_path = path_join(image_dir, image_file)
      # Get Attributes for each image
      value = extract_xml_contents(mat_path, img_path)
      # Append the attributes to the mat_list
      xml_list.append(value)

  # Columns for Pandas DataFrame
  column_name = ['filename', 'width', 'height', 'class_num', 'xmin', 'ymin',
                 'xmax', 'ymax']

  # Create the DataFrame from mat_list
  xml_df = pd_df(xml_list, columns=column_name)

  # Return the dataframe
  return xml_df

def preprocess_dataset():
  # Lists that will contain the whole dataset
  labels = []
  boxes = []
  img_list = []

  h = 256
  w = 256
  image_dir = 'dataset/images'

  with open('dataset.csv') as csvfile:
      rows = csv_reader(csvfile)
      columns = next(iter(rows))
      for row in rows:
        labels.append(int(row[3]))
        #Scaling Coordinates to the range of [0,1] by dividing the coordinate with image size, 256 here.
        arr = [float(row[4])/256,
               float(row[5])/256,
               float(row[6])/256,
               float(row[7])/256]
        boxes.append(arr)
        img_path = row[0]
        # Read the image
        img  = cv2.imread(path_join(image_dir, img_path))

        # Resize all images to a fix size
        image = cv2.resize(img, (256, 256))

        # # Convert the image from BGR to RGB as NasNetMobile was trained on RGB images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize the image by dividing it by 255.0
        image = image.astype("float") / 255.0

        # Append it to the list of images
        img_list.append(image)

  return labels, boxes, img_list

def get_example_figs(img_list, boxes, img_size = 256):
    # Create a Matplotlib figure
    plt.figure(figsize=(20, 20))
    # Generate a random sample of images each time the cell is run
    random_range = random.sample(range(1, len(img_list)), 20)

    for itr, i in enumerate(random_range, 1):
        # Bounding box of each image
        a1, b1, a2, b2 = boxes[i]

        # Rescaling the boundig box values to match the image size
        x1 = a1 * img_size
        x2 = a2 * img_size
        y1 = b1 * img_size
        y2 = b2 * img_size

        # The image to visualize
        image = img_list[i]

        # Draw bounding boxes on the image
        cv2.rectangle(image, (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0),
                      3)

        # Clip the values to 0-1 and draw the sample of images
        img = np_clip(img_list[i], 0, 1)
        plt.subplot(4, 5, itr)
        plt.imshow(img)
        plt.axis('off')

def split_data(img_list, boxes, labels, test_size=0.2, random_state=43):
    # Split the data of images, labels and their annotations
    train_images, val_images, train_labels, \
    val_labels, train_boxes, val_boxes = train_test_split(np_array(img_list),
                                                          np_array(labels), np_array(boxes), test_size=test_size,
                                                          random_state=random_state)

    print('Training Images Count: {}, Validation Images Count: {}'.format(len(train_images), len(val_images)))

    return train_images, val_images, train_labels, val_labels, train_boxes, val_boxes

def get_image_list(verbose=0):
    image_list = glob.glob('dataset/images/*.jpg')
    if verbose>0:
        print("image count = {}".format(len(image_list)))
    return image_list

def get_random_img(verbose=0):
    image_list = get_image_list(verbose=verbose-1)
    imageid = np_randint(len(image_list))
    if verbose>0:
        print("image with id({}) is {}".format(imageid, image_list[imageid]))
    return image_list[imageid]