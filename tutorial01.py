import localization_example_helper_funcs
from localization_example_helper_funcs import xml_to_csv as ex_xml2csv, preprocess_dataset as ex_preprocess_dataset
from localization_example_helper_funcs import split_data as ex_split_data
from localization_example_helper_funcs import Dataset as ex_Dataset, ValDataset as ex_ValDataset
import random
from helperFuncs import is_cuda_available, check_and_return_device
from localization_example_helper_funcs import Network as ex_Network, train as ex_train_network
from localization_example_helper_funcs import predict as ex_predict
from torch.utils.data import DataLoader as torchDataLoader
import matplotlib.pyplot as plt

def load_data():
    num_to_labels = {0: 'cat', 1: 'dog'}
    # The Classes we will use for our training
    classes_list = sorted(['cat',  'dog'])
    # Run the function to convert all the xml files to a Pandas DataFrame
    labels_df = ex_xml2csv()
    # Saving the Pandas DataFrame as CSV File
    labels_df.to_csv(('dataset.csv'), index=None)
    # All images will resized to 300, 300
    image_size = 256
    # Get Augmented images and bounding boxes
    labels, boxes, img_list = ex_preprocess_dataset()
    return labels, boxes, img_list

def preprocess_data(img_list, boxes, labels):
    # Now we need to shuffle the data, so zip all lists and shuffle
    combined_list = list(zip(img_list, boxes, labels))
    random.shuffle(combined_list)
    # Extract back the contents of each list
    img_list, boxes, labels = zip(*combined_list)
    train_images, val_images, train_labels, val_labels, train_boxes, val_boxes = ex_split_data(img_list, boxes, labels)
    dataset = ex_Dataset(train_images, train_labels, train_boxes)
    valdataset = ex_ValDataset(val_images, val_labels, val_boxes)
    dataloader = torchDataLoader(dataset, batch_size=32, shuffle=True)
    valdataloader = torchDataLoader(valdataset, batch_size=32, shuffle=True)
    return dataloader, valdataloader

def get_model(device):
    model = ex_Network()
    model = model.to(device)
    print(model)

def plot_predicted_img(img):
    # Show the Image with matplotlib
    plt.clf()
    f = plt.figure(figsize=(5, 5))
    plt.imshow(img[:, :, ::-1])
    plt.show()
    return f

def main():
    is_cuda_available()
    check_and_return_device()

    labels, boxes, img_list = load_data()
    dataloader, valdataloader = preprocess_data(img_list, boxes, labels)

    model, acc_list = ex_train_network(dataloader, valdataloader, model=None, num_of_epochs=30, start_from_scratch=False)

    rand_img_path = localization_example_helper_funcs.get_random_img()
    img = ex_predict(rand_img_path, model_epoch_id=30)
    plot_predicted_img(img)
