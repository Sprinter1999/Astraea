# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets

"""## Load Data"""

# 数据集导入
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# create transforms
# We will just convert to tensor and normalize
transforms_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Load Data
mnist_data_train = datasets.MNIST(
    './data/mnist/', train=True, download=True, transform=transforms_mnist)
mnist_data_test = datasets.MNIST(
    '../data/mnist/', train=False, download=True, transform=transforms_mnist)
classes = np.array(list(mnist_data_train.class_to_idx.values()))
classes_test = np.array(list(mnist_data_test.class_to_idx.values()))
num_classes = len(classes)
print("Classes: {} \tType: {}".format(classes, type(classes)))
print("Classes Test: {} \tType: {}".format(classes_test, type(classes)))
print("Image Shape: {}".format(mnist_data_train.data[0].size()))
print("Load Data Done!")
classes_mapper = {}
for item in classes:
    # numpy64转int
    classes_mapper[int(item)] = []

"""## Data Partitioning"""


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def classesIdxsGenerator(classes, dataset):
    # 这里可以使用pickle模块进行dump，不用每次都跑一遍
    for i in range(len(dataset)):
        # tensor转int
        targetForOnce = int(dataset.targets[i])
        classes_mapper[targetForOnce].append(i)

    for key in classes_mapper.keys():
        # list转为numpy数组，所以mapper的k-v类型是int-numpy_array
        classes_mapper[key] = np.array(classes_mapper[key])
    return classes_mapper


classes_mapper = classesIdxsGenerator(
    classes=classes, dataset=mnist_data_train)


def customizedIdxsGenerator(Bias_array, local_datasize):
    """
    @Params:
    Bias_array即用于表示该客户端数据集类别偏好的数组（可能要改为字典存储）
    local_datasize用于表示该客户端有多少样本量
    """
    # list为可变对象，需要深拷贝
    Bias_num_array = copy.deepcopy(Bias_array)
    for i in range(len(Bias_array)):
        Bias_num_array[i] = int(local_datasize*Bias_array[i])

    local_dataitem_idxs = np.array([])
    # classes_mapper=classesIdxsGenerator(classes=classes,dataset=mnist_data_train)

    for key in classes_mapper.keys():
        # 先把classes_mapper[key] list类型转化numpy
        classes_mapper[key] = np.array(classes_mapper[key])
        # concatenate
        local_dataitem_idxs = np.concatenate((local_dataitem_idxs, np.random.choice(
            classes_mapper[key], int(Bias_num_array[key]), replace=False)), axis=0)

    np.random.shuffle(local_dataitem_idxs)
    local_dataitem_idxs = local_dataitem_idxs.astype(int)
    return local_dataitem_idxs.tolist()


def iid_partition(dataset, numOfClients):

    num_items_per_client = int(len(dataset)/numOfClients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(numOfClients):
        client_dict[i] = set(np.random.choice(
            image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])
    return client_dict


def non_iid_partition(dataset, clients, total_shards, shards_size, num_shards_per_client):

    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}
    idxs = np.arange(len(dataset))
    data_labels = dataset.targets.numpy()

    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(clients):
        rand_set = set(np.random.choice(
            shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate(
                (client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)

    return client_dict


"""## LeNet Model"""


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution,padding=2 because MNIST is of 28*28
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""## Client"""


class Client(object):
    def __init__(self, client_index, dataset, batchsize, learning_rate, local_epochs, data_idxs, bias_array):
        self.client_index = client_index
        self.bias_array = torch.Tensor(bias_array)
        self.data_idxs = data_idxs
        self.train_loader = DataLoader(CustomDataset(
            dataset, data_idxs), batch_size=batchsize, shuffle=True)
        self.learning_rate = learning_rate
        self.epochs = local_epochs
        self.local_datasize = len(data_idxs)

    def printProfile(self):
        print(
            f"ClientNumber: {self.client_index},learningRate: {self.learning_rate},local_epochs {self.epochs},and its bias is {self.bias_array}")
        print(
            f"Type of dataidxs is {type(self.data_idxs)}, and they are {self.data_idxs}")
        return

    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        # Adam优化器可以自适应调整学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        e_loss = []
        for _ in range(1, self.epochs+1):  # _ means epoch
            train_loss = 0.0
            model.train()
            for data, labels in self.train_loader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)

            # average losses
            train_loss = train_loss/len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss)/len(e_loss)
        return model.state_dict(), total_loss


"""## Edge or Mediator"""


class Edge(object):
    def __init__(self, edge_index, region_clients_idxs, Cr):
        self.edge_index = edge_index
        self.region_clients_idxs = region_clients_idxs
        self.Cr = Cr
        self.Bias_array = torch.Tensor([0.0 for i in range(10)])
        # 建立client_index到对应weights和datasize的键值对，初始化占位即可
        self.cachedWeights, self.cachedDatasizes = {}, {}
        for i in range(len(self.region_clients_idxs)):
            self.cachedWeights[self.region_clients_idxs[i]] = 0
            self.cachedDatasizes[self.region_clients_idxs[i]] = 0

    def addClient(self, client_index):
        self.region_clients_idxs.append(client_index)
        return

    def printProfile(self):
        sumOfDistribution = torch.sum(self.Bias_array)
        print(f"Edge_index: {self.edge_index}, region clients: {self.region_clients_idxs},Cr: {self.Cr}, and its bias is {self.Bias_array}, which sums upto {sumOfDistribution}")
        return

    # edge端单次更新
    def edgeUpdate(self, model):
        self.selected_client_num = max(
            int(self.Cr*len(self.region_clients_idxs)), 1)
        self.selected_client_idxs = np.random.choice(
            self.region_clients_idxs, self.selected_client_num, replace=False)
        # print(f"Edge#{self.edge_index} this turn select {self.selected_client_idxs} ")
        self.tempTotalDatasize = 0
        local_Loss = []
        for k in self.selected_client_idxs:
            local_update = TotalClients[k]
            weights, _ = local_update.train(
                model=copy.deepcopy(model))  # _ 指本次loss
            local_Loss.append(copy.deepcopy(_))
            self.cachedWeights[k] = copy.deepcopy(weights)
            self.cachedDatasizes[k] = local_update.local_datasize
            self.tempTotalDatasize += local_update.local_datasize

        loss_avg = sum(local_Loss) / len(local_Loss)
        # print(f"Edge#{self.edge_index} this turn LossAvg is {loss_avg}, lenOf localLoss is{len(local_Loss)}")

    def edgeAggregation(self):
        weights_avg = copy.deepcopy(
            self.cachedWeights[self.selected_client_idxs[0]])
        for key in weights_avg.keys():
            weights_avg[key] *= self.cachedDatasizes[self.selected_client_idxs[0]
                                                     ]/self.tempTotalDatasize
        for key in weights_avg.keys():
            for k in self.selected_client_idxs[1:]:
                ratio = self.cachedDatasizes[k]/self.tempTotalDatasize
                # test print
                # if(self.edge_index==1):
                # 	print(' ',ratio,end='')
                weights_avg[key] += self.cachedWeights[k][key]*ratio
            # weights_avg[key]=torch.div(weights_avg[key], len(self.selected_client_idxs))

        return weights_avg, self.tempTotalDatasize


"""## HyBridFL protocol"""


def HyBridFL(model, rounds):
    global_weights = model.state_dict()
    # 全局更新轮数
    for i in range(rounds):
        print(f"\nGLOBAL ROUND {i}")
        temp_weights, temp_datasizes = [], []
        # 遍历每一个Edge
        for j in range(Edges):
            # 模型分发
            TotalEdges[j].edgeUpdate(model=copy.deepcopy(model))
            edge_weights, edge_datasize = TotalEdges[j].edgeAggregation()
            temp_weights.append(copy.deepcopy(edge_weights))
            temp_datasizes.append(edge_datasize)

        # Edge端聚合
        sumOfTempDatasizes = sum(temp_datasizes)
        weights_global_avg = copy.deepcopy(temp_weights[0])
        for key in weights_global_avg.keys():
            weights_global_avg[key] *= temp_datasizes[0]/sumOfTempDatasizes
        for key in weights_global_avg.keys():
            for k in range(1, len(temp_weights)):
                ratio = temp_datasizes[k]/sumOfTempDatasizes
                # print(f"Edge#{k} this turn datasize ratio {ratio} across all edges")
                weights_global_avg[key] += temp_weights[k][key]*ratio

            # weights_global_avg[key] = torch.div(weights_global_avg[key], len(temp_weights))

        global_weights = weights_global_avg
        print('\n')
        model.load_state_dict(weights_global_avg)
        Criterion = nn.CrossEntropyLoss()
        TestGlobalModel(model=model, dataset=mnist_data_test, criterion=Criterion,
                        test_batchsize=128, num_classes=num_classes, classes=classes)


"""## Testing"""

# TODO: Test Global Model


def TestGlobalModel(model, dataset, criterion, test_batchsize, num_classes, classes):
    # test loss
    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(dataset, batch_size=test_batchsize)
    l = len(test_loader)
    model.eval()
    for data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item()*data.size(0)

        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available(
        ) else np.squeeze(correct_tensor.cpu().numpy())

        # test accuracy for each object class
        for i in range(num_classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    # avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print("Test Loss: {:.6f}\n".format(test_loss))

    # print test accuracy
    for i in range(10):
        if total_class[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %
                  (classes[i], 100 * correct_class[i] / total_class[i],
                   np.sum(correct_class[i]), np.sum(total_class[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' %
                  (classes[i]))

    print('\nFinal Test  Accuracy: {:.3f} ({}/{})'.format(
        100. * np.sum(correct_class) / np.sum(total_class),
        np.sum(correct_class), np.sum(total_class)))

    test_accuracies.append(100. * np.sum(correct_class) / np.sum(total_class))


"""## Basic Experiment Settings"""

"""## 基本实验设置"""

#TODO: 基本实验设置
# global rounds
t_max = 100
# global client fraction
C = 0.1
# number of total clients
K = 500
# number of total edges
Edges = 10
# local_epochs
E = 5
# learning_rate&batchsize
lr = 0.0005
batchSize = 10
# local_datasize
local_datasize = len(mnist_data_train)/K

iid_dict = iid_partition(mnist_data_train, K)
TotalClients = []
Clients_all_idxs = [i for i in range(K)]
TotalEdges = []
ClientsNumForEdges = [0 for i in range(Edges)]
test_accuracies = []
BIASARRAY = [0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0, 0, 0]
BIASDISTRIBUTE = [
    [0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0, 0, 0],
    [0, 0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0, 0],
    [0, 0, 0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0],
    [0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0, 0, 0],
    [0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0, 0],
    [0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0],
    [0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4],
    [0.4, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3],
    [0.3, 0.4, 0, 0, 0, 0, 0, 0, 0.1, 0.2],
    [0.2, 0.3, 0.4, 0, 0, 0, 0, 0, 0, 0.1],
]

Set_clients = set(Clients_all_idxs)


def Rescheduling(C=0.1, Max_clients_per_mediator=10):
    # 初始化客户端群体
    from random import randint
    print(f"Generating {K} clients and sampling random generation profile...")
    for i in range(K):
        # print(f"For {i}")
        samplingIdx = i % num_classes
        biasArrayTemp = BIASDISTRIBUTE[samplingIdx]
        clientDatasetIdxs = customizedIdxsGenerator(
            Bias_array=biasArrayTemp, local_datasize=local_datasize)
        clientTemp = Client(client_index=i, dataset=mnist_data_train, batchsize=batchSize,
                            learning_rate=lr, local_epochs=E, data_idxs=clientDatasetIdxs, bias_array=biasArrayTemp)
        if(i % 48 == 0):
            clientTemp.printProfile()
        TotalClients.append(clientTemp)
    print(f"Done, {K} clients are generated...")

    edge_index_temp = 0
    P_uniform = torch.Tensor([0.1 for i in range(10)])
    # 只要还有clients就重复最外层循环
    while(len(Set_clients) != 0):
        # 初始化一个mediator
        mediator_temp = Edge(edge_index=edge_index_temp,
                             region_clients_idxs=[], Cr=C)
        P_mediator = copy.deepcopy(mediator_temp.Bias_array)  # Tensor类型的
        curClientsCount = 0
        # 寻找一个P_mediator本次的全部从属client
        while(len(Set_clients) != 0 and len(mediator_temp.region_clients_idxs) < Max_clients_per_mediator):
            # 先定一个临时散度最大值
            kl_div_min = 9999999
            P_mediator = copy.deepcopy(mediator_temp.Bias_array)  # Tensor类型的
            # 遍历每一个Set_clients的client找到合适的k
            target_K = 1
            # 寻找一次mediator本轮中的一个client
            for client_idx in Set_clients.copy():
                P_client = copy.deepcopy(TotalClients[client_idx].bias_array)
                # 如果当前mediator还没有clients,就先给它分配当前的client
                if len(mediator_temp.region_clients_idxs) == 0:
                    print(
                        f"Current MediatorIndex is: {edge_index_temp} ,and first allocate client_index:{client_idx} ,and its bias is:{P_client}")
                    mediator_temp.addClient(client_idx)
                    Set_clients.remove(client_idx)
                    curClientsCount += 1
                    P_uniform += torch.Tensor([0.1 for i in range(10)])
                    P_mediator = copy.deepcopy(P_client)
                    continue
                else:
                    P_fusion_temp = P_client+P_mediator  # 论文中并没有提到具体怎么做的
                    # 想要计算D(p||q)，要写成kl_div（q.log（），p）,有点别扭
                    kl_div_temp = F.kl_div(
                        P_uniform.log(), P_fusion_temp, reduction='sum')/(curClientsCount+1)
                    if(kl_div_temp < kl_div_min):
                        kl_div_min = kl_div_temp
                        target_K = client_idx

            # 找到本轮的client k之后，更新当前mediator的各项参数
            mediator_temp.addClient(target_K)
            curClientsCount += 1
            P_client = TotalClients[target_K].bias_array
            P_fusion = P_mediator + P_client
            mediator_temp.Bias_array = copy.deepcopy(P_fusion)
            # 按次调整P_uniform
            P_uniform += torch.Tensor([0.1 for i in range(10)])
            # 从尚未匹配到的客户端Set中移除target_K
            Set_clients.remove(target_K)
        # 找完一个mediator的全部client之后
        mediator_temp.printProfile()
        edge_index_temp += 1
        TotalEdges.append(mediator_temp)


def main():
    # load model
    lenet = LeNet()
    if torch.cuda.is_available():
        lenet.cuda()
    Rescheduling(C=C, Max_clients_per_mediator=10)
# 后面先不管
    HyBridFL(model=lenet, rounds=100)
    print(test_accuracies)


main()
