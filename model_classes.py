import torch.nn
from torch.nn import Linear, Sequential, Conv2d
from torch.nn import ReLU, MaxPool2d
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Softmax
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import save as torch_save
from torch import load as torch_load
from matplotlib import pyplot as plt
import rotnetfuncs as rnf

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score

from rotnetfuncs import angle_error, angle_difference

class Flatten(Module):
    '''
    This model is for conv autoencoders
    At some point the convolution layers turn to Linear layers
    This layer will be used for flattening and unflattening such layers
    '''
    def __init__(self):
        super(Flatten, self).__init__()
        self.in_size = None

    def forward(self, input):
        if self.in_size is None:
            self.in_size = list(np.shape(input)[1:]) #[input.size(1), input.size(2), input.size(3)]
        return input.view(input.size(0), -1)

    def backward(self, input):
        return input.view(tuple([input.size(0)]) + tuple(self.in_size)) # input.view(input.size(0), self.in_size[0], self.in_size[1], self.in_size[2])

    def flatten(self, input):
        return self.forward(input)

    def unflatten(self, input):
        return self.backward(input)

class MLP(Module):
    # define model elements
    def __init__(self, nb_filters=[32, 64], kernel_size=[5, 5], hidCounts=[128,32], classCount=10, rot_deg_inc=1, network_id=0):
        super(MLP, self).__init__()
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.hidCounts = hidCounts
        self.classCount = classCount
        self.layerList = torch.nn.ModuleList()
        self.rot_deg_inc = rot_deg_inc
        self.network_id = network_id
        self.assign_network()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("model device will be= ", self.device)
        self.to(self.device)

    def assign_network(self):
        if self.network_id==0:
            self.layerList.append(
                Sequential(
                    Conv2d(in_channels=1, out_channels=self.nb_filters[0], kernel_size=self.kernel_size[0], stride=1, padding=2),
                    ReLU(),
                    MaxPool2d(kernel_size=2)))
            self.layerList.append(torch.nn.Dropout(0.25))
            self.layerList.append(
                Sequential(
                    Conv2d(in_channels=self.nb_filters[0], out_channels=self.nb_filters[1], kernel_size=self.kernel_size[1], stride=1,
                           padding=2),
                    ReLU(),
                    MaxPool2d(kernel_size=2)))
            self.layerList.append(torch.nn.Dropout(0.25))
            self.layerList.append(Flatten())
            # input to first hidden layer
            self.layerList.append(Linear(self.nb_filters[1] * 7 * 7, self.hidCounts[0]))
            self.layerList.append(ReLU())
            self.layerList.append(Linear(self.hidCounts[0], self.hidCounts[1]))
            self.layerList.append(ReLU())
            self.layerList.append(Linear(self.hidCounts[1], self.classCount))
            kaiming_uniform_(self.layerList[5].weight, nonlinearity='relu')
            kaiming_uniform_(self.layerList[7].weight, nonlinearity='relu')
            xavier_uniform_(self.layerList[9].weight)
        elif self.network_id==1:
            self.layerList.append(
                Sequential(
                    Conv2d(in_channels=1, out_channels=self.nb_filters[0], kernel_size=self.kernel_size[0], stride=1, padding=0),
                    ReLU()))
            self.layerList.append(torch.nn.Dropout(0.25))
            self.layerList.append(
                Sequential(
                    Conv2d(in_channels=self.nb_filters[0], out_channels=self.nb_filters[1], kernel_size=self.kernel_size[1], stride=1, padding=0),
                    ReLU(),
                    MaxPool2d(kernel_size=2)))
            self.layerList.append(torch.nn.Dropout(0.25))
            self.layerList.append(Flatten())
            # input to first hidden layer
            self.layerList.append(Linear(self.nb_filters[1]*12*12, self.hidCounts[0]))
            self.layerList.append(ReLU())
            self.layerList.append(torch.nn.Dropout(0.25))
            self.layerList.append(Linear(self.hidCounts[0], self.classCount))
            kaiming_uniform_(self.layerList[5].weight, nonlinearity='relu')
            xavier_uniform_(self.layerList[8].weight)
        return
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        for layer in self.layerList:
            #print("*****\nLayer info:\n", layer)
            #print("Before layer size= {}".format(X.size()))
            X = layer(X)
            #print("After layer size= {}\n*****".format(X.size()))
        return X

    # train the model
    def train_model(self, train_dl, batch_size=128, epochCnt=500):
        self.train()
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=0.01) #optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)

        train_dataloader = DataLoader(train_dl, batch_size=batch_size, shuffle=True)
        n = len(train_dl)

        time_dict = {}
        time_dict["train_start"] = datetime.now()

        # enumerate epochs
        for epoch in range(epochCnt):
            # enumerate mini batches
            time_dict["epoch_start"] = datetime.now()
            time_dict["batch_n"] = 0
            angleErrList = []
            angleErrAccum = 0
            epochLoss = 0
            for i, samples in enumerate(train_dataloader):
                inputs, targets, idx = samples['image'].to(self.device), samples['label'].long().to(self.device), samples['id']
                n = len(idx)
                inputs = inputs.reshape(n, 1, 28, 28).float()
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
                time_dict["batch_n"] = time_dict["batch_n"] + 1
                epochLoss += loss.item()
                angleErr, angErrVec = angle_error(targets.cpu(), yhat.detach().cpu(), self.rot_deg_inc)
                angleErrList.append(angErrVec)
                angleErrAccum += angleErr
            acc = self.evaluate_model(train_dl, batch_size=batch_size)

            epochLoss = epochLoss/n
            angleErrAccum /= n
            time_dict["epoch_dif"] = datetime.now()-time_dict["epoch_start"]
            time_dict["epoch_mean"] = time_dict["epoch_dif"]/time_dict["batch_n"]
            acc_dif_list = [torch.min(torch.cat(angleErrList)), torch.mean(torch.cat(angleErrList).float()), torch.max(torch.cat(angleErrList)), torch.sum((torch.cat(angleErrList)<5)), torch.sum((torch.cat(angleErrList)<10))]
            print('Epoch [{}/{}], acc: {:.4f}, Loss: {:.4f}, angErrMean {:4.2f}, elapsed {:4.2f} ms, perBatch {:4.2f} ms'.format(epoch + 1, epochCnt, acc, epochLoss, angleErrAccum, 1000*time_dict["epoch_dif"] .total_seconds(), 1000*time_dict["epoch_mean"] .total_seconds()))
            print("({:6d})<5,({:6d})<10,avg({}),min({}),max({})".format(acc_dif_list[3], acc_dif_list[4], acc_dif_list[1], acc_dif_list[0], acc_dif_list[2]))

        time_dict["train_dif"] = datetime.now() - time_dict["train_start"]
        print('Epoch [{}/{}], Loss: {:.4f}, total elapsed {:4.2f} ms'.format(epoch + 1, epochCnt, epochLoss, 1000 * time_dict["train_dif"].total_seconds()))

    # evaluate the model
    def evaluate_model(self, test_dl, batch_size=64):
        self.eval()
        test_dataloader = DataLoader(test_dl, batch_size=batch_size, shuffle=False)
        predictions, actuals = list(), list()
        sm = Softmax(dim=1)
        for i, samples in enumerate(test_dataloader):
            inputs, targets, idx = samples['image'].to(self.device), samples['label'].long().to(self.device), samples['id']
            n = len(idx)
            inputs = inputs.reshape(n, 1, 28, 28).float()
            # evaluate the model on the test set
            yhat = self.forward(inputs)
            yhat = sm(yhat)
            # retrieve numpy array

            yhat = yhat.cpu().detach().numpy()
            actual = targets.cpu().numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        # calculate accuracy
        # diff = angle_difference(np.argmax(y_true, axis=1) * rot_deg_inc, y_pred * rot_deg_inc)
        angleErr, angErrVec = angle_error(actuals.squeeze(), predictions, self.rot_deg_inc)
        acc_dif_list = [np.min(angErrVec), np.mean(angErrVec), np.max(angErrVec), np.sum(angErrVec<5), np.sum(angErrVec<10)]

        idx = np.argsort(angErrVec)
        angErrVecSorted = self.rot_deg_inc*angErrVec[idx]
        select_to_display = np.zeros((7,1), dtype=np.uint16).squeeze()
        select_to_display[0] = idx[0]
        for i in range(1, 5):
            select_to_display[i] = idx[np.argmax(angErrVecSorted>(i-1)*self.rot_deg_inc)]
        select_to_display[5] = idx[np.argmax(angErrVecSorted>10)]
        select_to_display[6] = idx[np.argmax(angErrVecSorted>20)]
        print(select_to_display, self.rot_deg_inc*angErrVec[select_to_display])
        print("({:6d})<5,({:6d})<10,avg({}),min({}),max({})".format(acc_dif_list[3], acc_dif_list[4], acc_dif_list[1], acc_dif_list[0], acc_dif_list[2]))

        mnist_show_rotated_examples(test_dl, select_to_display, self.rot_deg_inc, predictions)

        acc = accuracy_score(actuals, predictions)
        return acc

    def train_evaluate_trvate(self, train_dl, valid_dl, test_dl, epochCnt=500):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        accvectr = np.zeros(epochCnt)
        accvecva = np.zeros(epochCnt)
        accvecte = np.zeros(epochCnt)
        for epoch in range(epochCnt):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
            acc_tr = self.evaluate_model(train_dl)
            acc_va = self.evaluate_model(valid_dl)
            acc_te = self.evaluate_model(test_dl)
            print("epoch ", epoch, "tr: %.3f" % acc_tr, "va: %.3f" % acc_va, "te: %.3f" % acc_te)
            accvectr[epoch] = acc_tr
            accvecva[epoch] = acc_va
            accvecte[epoch] = acc_te
        return accvectr, accvecva, accvecte

def mnist_show_rotated_examples(the_data, idx, rot_deg_inc, predictions):
    figure = plt.figure(figsize=(15, 12))
    n = len(idx)
    cols, rows = 3, n
    for i in range(1, n):
        sample_idx = idx[i]
        sample = the_data[sample_idx]
        im = sample["image"].reshape(28, 28)
        rotation_angle = sample["label"]*rot_deg_inc
        rotPredict = predictions[idx[i]]*rot_deg_inc
        original_image = rnf.generate_rotated_image(im, 360-(rotation_angle), size=(28, 28),
                                                   crop_center=False, crop_largest_rect=False)
        predicted_image = rnf.generate_rotated_image(im, 360-rotation_angle+rotPredict, size=(28, 28),
                                                   crop_center=False, crop_largest_rect=False)

        figure.add_subplot(rows, cols, i*cols+1)
        plt.title("Original")
        plt.axis("off")
        plt.imshow(original_image, cmap="gray")

        figure.add_subplot(rows, cols, i*cols+2)
        plt.title("Rotated {:}".format(rotation_angle))
        plt.axis("off")
        plt.imshow(im, cmap="gray")

        figure.add_subplot(rows, cols, i*cols+3)
        plt.title("P({:}),dif({})".format(rotPredict,angle_difference(rotation_angle, rotPredict)))
        plt.axis("off")
        plt.imshow(predicted_image, cmap="gray")


    plt.show()
    plt.close()