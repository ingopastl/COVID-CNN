import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from AlexNet import AlexNet
from VGG import VGG


def load_data():
    transform_config = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load set data
    train_data = datasets.ImageFolder(r'D:\Projects\Python\COVID-CNN\Covid19-dataset_Augmented\train',
                                      transform=transform_config)
    test_data = datasets.ImageFolder(r'D:\Projects\Python\COVID-CNN\Covid19-dataset_Augmented\test',
                                     transform=transform_config)

    valid_set_percentage = 0.2

    train_file_count = len(train_data)
    train_file_indexes = list(range(train_file_count))
    np.random.shuffle(train_file_indexes)

    # Separates training set from validation set
    split = int(np.floor(valid_set_percentage * train_file_count))
    train_indexes = train_file_indexes[split:]
    valid_indexes = train_file_indexes[:split]

    train_sampler = SubsetRandomSampler(train_indexes)
    valid_sampler = SubsetRandomSampler(valid_indexes)

    # Number of processes that generate batches in parallel
    num_workers = 0

    # Denotes the number of samples contained in each generated batch
    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


def train_network(model, criterion, optimizer, train_loader, valid_loader):
    num_epochs = 30

    min_valid_loss = np.Inf
    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        # Prepares to train
        model.train()
        for data, target in train_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Runs the image through the neural network
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        # Prepares to evaluate
        model.eval()
        for data, target in valid_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Runs the image through the neural network
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation loss: {:.4f}'.format(
            epoch, train_loss, valid_loss))

        if valid_loss <= min_valid_loss:
            print('Validation loss decreased ({:.4f} -> {:.4f}) \n Saving model...'.format(
                min_valid_loss, valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            min_valid_loss = valid_loss


def test_network(model, test_loader):
    correct = 0
    total = 0

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)
        _, predicted = torch.max(output, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':
    trainLoader, validLoader, testLoader = load_data()

    vgg_adam_Average = 0
    alex_adam_Average = 0
    vgg_RMSprop_Average = 0
    alex_RMSprop_Average = 0

    for i in range(0, 5):
        # VGG neural network
        vggModel = VGG(15)
        if torch.cuda.is_available():
            vggModel.cuda()
        # AlexNet neural network
        alexNetModel = AlexNet(15)
        if torch.cuda.is_available():
            alexNetModel.cuda()

        vggCriterion = nn.CrossEntropyLoss()
        alexNetCriterion = nn.CrossEntropyLoss()

        # Adam optimizer
        vggOptimizer = optim.Adam(vggModel.parameters(), lr=0.0001)
        alexOptimizer = optim.Adam(alexNetModel.parameters(), lr=0.0001)

        train_network(vggModel, vggCriterion, vggOptimizer, train_loader=trainLoader, valid_loader=validLoader)
        vggModel.load_state_dict(torch.load('model.pt'))
        vgg_adam_Average += test_network(vggModel, testLoader)
        os.rename('model.pt', str(i) + '_vgg_adam_model.pt')

        train_network(alexNetModel, alexNetCriterion, alexOptimizer, train_loader=trainLoader, valid_loader=validLoader)
        alexNetModel.load_state_dict(torch.load('model.pt'))
        alex_adam_Average += test_network(alexNetModel, testLoader)
        os.rename('model.pt', str(i) + '_alexNet_adam_model.pt')

        # RMSprop optimizer
        vggOptimizer = optim.RMSprop(vggModel.parameters(), lr=0.001)
        alexOptimizer = optim.RMSprop(alexNetModel.parameters(), lr=0.001)

        train_network(vggModel, vggCriterion, vggOptimizer, train_loader=trainLoader, valid_loader=validLoader)
        vggModel.load_state_dict(torch.load('model.pt'))
        vgg_RMSprop_Average += test_network(vggModel, testLoader)
        os.rename('model.pt', str(i) + '_vgg_RMSprop_model.pt')

        train_network(alexNetModel, alexNetCriterion, alexOptimizer, train_loader=trainLoader, valid_loader=validLoader)
        alexNetModel.load_state_dict(torch.load('model.pt'))
        alex_RMSprop_Average += test_network(alexNetModel, testLoader)
        os.rename('model.pt', str(i) + '_alexNet_RMSprop_model.pt')

    vgg_adam_Average = vgg_adam_Average / 5
    vgg_RMSprop_Average = vgg_RMSprop_Average / 5
    alex_adam_Average = alex_adam_Average / 5
    alex_RMSprop_Average = alex_RMSprop_Average / 5

    print("VGG - Adam: " + str(vgg_adam_Average))
    print("VGG - RMSprop: " + str(vgg_RMSprop_Average))
    print("Alex - Adam: " + str(alex_adam_Average))
    print("Alex - RMSprop: " + str(alex_RMSprop_Average))
