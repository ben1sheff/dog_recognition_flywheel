import torch
# import torchvision.transforms as transforms
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn

import stance_classifier as sc
# TO DO: this currenty just draws a train vs. validation plot to
# judge epochs of training needed. This is better done with callback
# functions, which I'm sure pyTorch supports
# Parameters
batch_size = 20
frac_valid = 0.2
training_epochs = 15
model_param_file = "model_parameters"
pictures_file = "stream_data.hdf5"  # "pictures.hdf5"  # location of the data
second_file = "pictures.hdf5"


def imshow(img):
    fig = plt.figure()
    img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    plt.imshow(img[0])
    plt.show(block=False)

f = h5py.File(pictures_file, "r")
f2 = h5py.File(second_file, "r")
dset = np.array(f["pics"], dtype="float32")
dset = np.concatenate((dset, np.array(f2["pics"], dtype="float32")))
tags = np.concatenate((np.array(f["sitting"], dtype="float32"), np.array(f2["sitting"], dtype="float32")))
tags = torch.from_numpy(tags)
dset = list(zip(dset, tags))
train, valid = torch.utils.data.random_split(dset, [1-frac_valid, frac_valid])
print("Training on", len(train), "data points, validating on", len(valid), "points")
# valid = dset[len(dset)*2//3:len(dset)*4//5]
# dset = dset[:len(dset)*2//3] + dset[len(dset)*4//5:]

# imshow(sc.TestConvolution(3)([dset[4]], 0))
# imshow(sc.TestConvolution(3)([dset[4]], 1))
# imshow(sc.TestConvolution(3)([dset[4]], 2))
# plt.show()

classifier = sc.StanceClassifier((512, 512))
print("Should we load the model? (y/n)")
if input().startswith("y"):
    classifier.load_state_dict(torch.load(model_param_file))
    valid = dset
else:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    # optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    # Prepping validation data
    val_data = torch.Tensor(np.array([val[0] for val in valid]))
    val_stance = torch.Tensor(np.array([val[1] for val in valid]))
    full_data = torch.Tensor(np.array([val[0] for val in train]))
    full_stance = torch.Tensor(np.array([val[1] for val in train]))
    losses = []
    val_losses = []
    for epoch in range(training_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(inputs)
            # print(labels, outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 10 == 9:    # print every 2000 mini-batches
            print(f'\n[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        losses += [criterion(classifier(full_data), full_stance).item()]
        val_losses += [criterion(classifier(val_data), val_stance).item()]  #  * (1 - frac_valid)/frac_valid]
    print('Finished Training')

    plt.plot(list(range(len(losses))), losses, label="raw loss")
    plt.plot(list(range(len(val_losses))), val_losses, label = "validation")
    plt.legend()
    plt.show()
# print(classifier(torch.tensor([dog[0] for dog in dset])))
tot = len(valid)
correct = 0
sit_guesses = 0
true_sits = 0
identified_sits = 0
for dog, tag in valid:
    result = classifier(torch.Tensor(dog[None,:,:]))
    if (result.item() > 0.5) == (tag.item() > 0.5):
        correct += 1
    if result.item() > 0.5:
        sit_guesses += 1
    if tag.item() > 0.5:
        true_sits += 1
        if result.item() > 0.5:
            identified_sits += 1
    # print(np.round(result.item(), 2), result.item()>0.5, "| Actual Tag:", tag.item()>0.5)
print(correct/tot * 100, "percent correct")
print("guessed sit", np.round(sit_guesses / tot * 100.), "percent of the time")
print("caught sitting", np.round(identified_sits / true_sits * 100.), "percent of the time")
print("caught standing", np.round((correct - identified_sits) / (tot-true_sits) * 100.), "percent of the time")
if len(valid) < len(dset):
    print("Should we save this? (y/n)")
    if input().startswith("y"):
        torch.save(classifier.state_dict(), model_param_file)
