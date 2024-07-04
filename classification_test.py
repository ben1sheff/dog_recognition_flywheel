import torch
import torchvision.transforms as transforms
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn

import stance_classifier as sc

# Parameters
batch_size = 4


def imshow(img):
    fig = plt.figure()
    img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    plt.imshow(img[0])
    plt.show(block=False)

f = h5py.File("pictures.hdf5", "r")
dset = np.array(f["pics"], dtype="float32")
tags = torch.from_numpy(np.array(f["sitting"], dtype="float32"))
dset = list(zip(dset, tags))

# imshow(sc.TestConvolution(3)([dset[4]], 0))
# imshow(sc.TestConvolution(3)([dset[4]], 1))
# imshow(sc.TestConvolution(3)([dset[4]], 2))
# plt.show()

classifier = sc.StanceClassifier((512, 512))
criterion = nn.BCELoss()
# optimizer = optim.Adam(classifier.parameters(), lr=0.01)
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs)
        print(labels, outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if True: # i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'\n[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
# print(classifier(torch.tensor([dog[0] for dog in dset])))
for dog, tag in zip(dset, tags):
    result = classifier(torch.Tensor(dog[0][None,:,:]))
    print(result.item()>0.5, tag.item()>0.5)

