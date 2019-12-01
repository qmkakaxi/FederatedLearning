
import matplotlib
import argparse
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from models.Update import LocalUpdate
import torch

from utils.sampling import mnist_iid
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

from torch.utils.data import DataLoader
import pickle
import random
from PIL import Image





def noisy_label_change_client(dataName, dict_users, dataset, noisy_client, noisy_rate):
    """
    change correct label into noisy label
    dataName:'mnist' or 'cifar'
    dict_users:每个用户分配的数据
    dataset:对应数据集的所有数据
    noisy_client:有噪音标签的用户数量
    noisy_rate:每个用户有多少比例的数据是错误的

    将一部分client的数据的标签置换为错误标签，选取前k个client
    不需要返回值，直接修改了原始数据集，故调用后，如需原始数据集，需要再次读取
    """
    if dataName == 'mnist':
        originTargets = dataset.train_labels.numpy()
    else:
        originTargets = dataset.targets
    allorigin_targets = set(originTargets)

    if noisy_client > len(dict_users):
        print('too many noisy client')
        raise NameError('noisy_client')
        exit()
    noisyDataList = []
    for userIndex in range(noisy_client):
        noisyDataList.extend(list(
            np.random.choice(list(dict_users[userIndex]), int(len(dict_users[userIndex]) * noisy_rate), replace=False)))

    for index in noisyDataList:
        all_targets = allorigin_targets
        all_targets = all_targets - set([originTargets[index]])
        new_label = np.random.choice(list(all_targets), 1, replace=False)
        originTargets[index] = new_label[0]
    dataset.targets = torch.tensor(originTargets)
    return dataset, noisyDataList,torch.tensor(originTargets)


def noisy_label( dataset,noisyDataList ):

    originTargets = dataset.train_labels.numpy()
    allorigin_targets = set(originTargets)

    for index in noisyDataList:
        all_targets = allorigin_targets
        all_targets = all_targets - set([originTargets[index]])
        new_label = np.random.choice(list(all_targets), 1, replace=False)
        originTargets[index] = new_label[0]
    dataset.targets = torch.tensor(originTargets)
    return dataset

def noisy_label_(dataset ,noisyDataList , newTargets):

    originTargets = dataset.train_labels.numpy()

    for index in noisyDataList:
        originTargets[index] = newTargets[index]
    dataset.targets = torch.tensor(originTargets)
    return dataset

def noisy_label_cifar(dataset ,noisyDataList , newTargets):

    originTargets = dataset.train_labels

    for index in noisyDataList:
        originTargets[index] = newTargets[index]
    dataset.targets = torch.tensor(originTargets)

    return dataset


#高斯噪声
def Gaussian(src,mu,sigma):
    NoiseImg=src
    NoiseNum=len(src[0])
    for i in range(NoiseNum):
        for j in range(NoiseNum):
            NoiseImg[i][j]=NoiseImg[i][j]+random.gauss(mu,sigma)
    return NoiseImg


#椒盐噪声
# def PepperandSalt(src,percetage):
#     NoiseImg=src
#     NoiseNum=int(percetage*src.shape[0]*src.shape[1])
#     for i in range(NoiseNum):
#         randX=random.random_integers(0,src.shape[0]-1)
#         randY=random.random_integers(0,src.shape[1]-1)
#         # if random.random_integers(0,1)<=0.5:
#         #     NoiseImg[randX,randY]=0
#         # else:
#         #     NoiseImg[randX,randY]=255
#     return NoiseImg



def Gaussian_noise(dataset,mu,sigma):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    num=len(dataset)
    datasets=[]
    for i in range(num):
        data=[]
        clean_image=np.array(dataset[i][0])
        noisy_image=Gaussian(clean_image,mu,sigma)
        noisy_image=Image.fromarray(noisy_image)
        noisy_image=trans_mnist(noisy_image)
        data.append(noisy_image)
        data.append(dataset[i][1])
        datasets.append(data)
    return datasets


def Gaussian_noise_dict(dataset,mu,sigma, dict_user, rate,num_client):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    datasets=list(dataset)
    num_noisy_client=int(rate*num_client)
    num_noisy=int(0.3*len(dict_user[0]))
    for i in range(num_noisy_client):
        dict=list(dict_user[i])
        for j in  range(num_noisy):
            data=[]
            clean_image=np.array(dataset[dict[j]][0])
            noisy_image = Gaussian(clean_image, mu, sigma)
            noisy_image = Image.fromarray(noisy_image)
            noisy_image = trans_mnist(noisy_image)
            data.append(noisy_image)
            data.append(dataset[dict[j]][1])
            datasets[dict[j]]=data

    for i in range(num_noisy_client):
        dict=list(dict_user[i])
        for j in  range(len(dict_user[0])-num_noisy):
            data=[]
            noisy_image = trans_mnist(dataset[dict[j+num_noisy]][0])
            data.append(noisy_image)
            data.append(dataset[dict[j+num_noisy]][1])
            datasets[dict[j+num_noisy]]=data

    for i in range(num_client-num_noisy_client):
        for _, index in enumerate(dict_user[i+num_noisy_client]):
            data=[]
            noisy_image = trans_mnist(dataset[index][0])
            data.append(noisy_image)
            data.append(dataset[index][1])
            datasets[index]=data
    return datasets


def Gaussian_noise_cifar(dataset,mu,sigma):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    num=len(dataset)
    datasets=[]
    for i in range(num):
        data=[]
        clean_image=np.array(dataset[i][0])
        noisy_image=Gaussian(clean_image,mu,sigma)
        noisy_image=Image.fromarray(noisy_image)
        noisy_image=transform(noisy_image)
        data.append(noisy_image)
        data.append(dataset[i][1])
        datasets.append(data)
    return datasets
