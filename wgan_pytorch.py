import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import cmnist
import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D

#need to process our data set instead
#mnist = cmnist.read_data_sets('../../MNIST_data', one_hot=True)
mnist = transform.data_importer_GAN(one_hot=True)
#parameter values have been changed to match with IDSGAN paper
mb_size = 64
z_dim = 9
X_dim = mnist.train.samples.shape[1]
#X_dim = mnist.train.images.shape[1]

y_dim = mnist.train.labels.shape[1]

h_dim = 128
cnt = 0
lr = 1e-4
epoch=100

G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    # torch.nn.Linear(h_dim, X_dim),
    # torch.nn.ReLU(),
    # torch.nn.Linear(h_dim, X_dim),
    # torch.nn.ReLU(),
    # torch.nn.Linear(h_dim, X_dim),
    # torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
)

def reset_grad():
    G.zero_grad()
    D.zero_grad()


G_solver = optim.RMSprop(G.parameters(), lr=lr)
D_solver = optim.RMSprop(D.parameters(), lr=lr)

#In IDS.py, need to train IDS beforehand using half of training set and save parameters
#Other half of training set to be used here, need to save parameters of G, D after finishing iteration-done
#Here need to load saved parameters of IDS before the loop
# train_data= mnist.train.samples.values.reshape(mnist.train.samples.shape[0], mnist.train.samples.shape[1], 1)
# IDS=Sequential()
# IDS.add(Conv1D(32, 3, input_shape=train_data.shape[1:]))
# IDS.add(Activation('relu'))
# IDS.add(Conv1D(64, 3))
# IDS.add(Activation('relu'))
# IDS.add(MaxPooling1D(pool_size=2))
# IDS.add(LSTM(70, dropout=0.1))
# IDS.add(Dropout(0.1))
# IDS.add(Dense(70))
# IDS.add(Activation('softmax'))
# IDS.load_weights("models/ids/weight.h5")
for it in range(epoch):
    for _ in range(5):
        # Sample data

        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X.values))
        #X = Variable(torch.from_numpy(X))

        # Dicriminator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, z_dim))
        G_sample = G(z)
        # G_sample = G_sample.detach().numpy()
        # sample_data = G_sample.reshape(G_sample.shape[0], G_sample.shape[1], 1)
        # X_data = X.reshape(X.shape[0], X.shape[1], 1)
        # X_data = X_data.detach().numpy()
        #classify G_sample and X by IDS here
        # I_sample=IDS.predict(sample_data,batch_size=mb_size)
        # I_sample = Variable(torch.from_numpy(I_sample))
        # I_real=IDS.predict(X_data,batch_size=mb_size)
        # I_real = Variable(torch.from_numpy(I_real))
        #Use I_sample and I_real as input for D
        # D_real=D(I_real)
        # D_fake=D(I_sample)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

        D_loss.backward()
        D_solver.step()

        # Weight clipping
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Housekeeping - reset gradient
        reset_grad()

    # Generator forward-loss-backward-update
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X.values))
    #X = Variable(torch.from_numpy(X))
    z = Variable(torch.randn(mb_size, z_dim))

    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = -torch.mean(D_fake)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if it % 10 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'
              .format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        # samples = G(z).data.numpy()[:16]
        #
        # fig = plt.figure(figsize=(4, 4))
        # gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.05, hspace=0.05)

        #for i, sample in enumerate(samples):
            #ax = plt.subplot(gs[i])
            #plt.axis('off')
            #ax.set_xticklabels([])
            #ax.set_yticklabels([])
            #ax.set_aspect('equal')
            #plt.imshow(sample.reshape(28, 28), cmap='Greys_r')*/

        if not os.path.exists('out/'):
            os.makedirs('out/')

        #plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        #plt.close(fig)

torch.save(G, 'G_model.pth')
torch.save(D, 'D_model.pth')