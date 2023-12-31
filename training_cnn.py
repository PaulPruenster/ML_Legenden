import os
import numpy as np
import dataset
from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from CNN import CNN
from FullyConnected import FullyConnected

# Setting the path of the training dataset (that was already provided to you)
running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
DATASET_PATH = "."

# Set the location of the dataset
if running_local:
    # If running on your local machine, the sign_lang_train folder's path should be specified here
    local_path = "sign_lang_train"
    if os.path.exists(local_path):
        DATASET_PATH = local_path
else:
    # If running on the Jupyter hub, this data folder is already available
    # You DO NOT need to upload the data!
    DATASET_PATH = "/data/mlproject22/sign_lang_train"

# importing the dataset
sign_lang_dataset = dataset.SignLangDataset(csv_file="labels.csv", root_dir=DATASET_PATH)

trainloader = DataLoader(sign_lang_dataset,
                         batch_size=64,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0)

PATH_CNN = "./net/CNN_50_epoch.pt"
PATH_FULLY_CONNECTED = "./net/FullyConnected_50_epochen.pt"

cnn = CNN()
fully_connected = FullyConnected()

# Choose what model to train
path_trained_net = PATH_CNN
model = cnn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if torch.cuda.is_available():
    print("using cuda")
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

epoch = 0
loss_arr = []
iter_counter = 0

# first over fit the net, then with decreasing amounts of training the net gets well-fitted
for n_epochs in range(50, 1, -5):
    print("ok")
    dataiter = iter(trainloader)
    for data_dict_train in dataiter:
        train_y = data_dict_train["label"]
        train_x = data_dict_train["image"]
        train_x = train_x / 255.0

        epoch += 1
        running_loss = 0.0
        for i in range(n_epochs):
            # getting the training set
            inputs, labels = Variable(train_x), Variable(train_y)

            # converting the data into GPU format
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # add the loss to an array for the plot
            #loss_arr.append(loss.item())

            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


# validation batch
data_dict_val = next(iter(trainloader))
val_x = data_dict_train["image"]
val_x = val_x / 255.0
val_y = data_dict_train["label"]

model.cpu()
torch.save(model.state_dict(), path_trained_net)

with torch.no_grad():
    output = model(train_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
print("accuracys: training set and validation set")
print(accuracy_score(train_y, predictions))

# prediction for validation set
with torch.no_grad():
    output = model(val_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on validation set
print(accuracy_score(val_y, predictions))

# Write the array to a file
#print("writing loss array to file...")
#with open('loss_CCN_50_epochen.txt', 'w') as file:
#    for item in loss_arr:
#        file.write(str(item) + '\n')
#print("done")