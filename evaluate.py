#We shall use test set here. Generator G will take test sample as input and give fake malicious sample as output.
#IDS would try to classify this fake sample
import transform
import torch
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D
from torch.autograd import Variable

data = transform.data_importer_IDS()
train_data= data.train.samples.values.reshape(data.train.samples.shape[0], data.train.samples.shape[1], 1)
test_data= data.test.samples.values.reshape(data.test.samples.shape[0], data.test.samples.shape[1], 1)

print(data.test.labels)

IDS=Sequential()
IDS.add(Conv1D(32, 3, input_shape=train_data.shape[1:]))
IDS.add(Activation('relu'))
IDS.add(Conv1D(64, 3))
IDS.add(Activation('relu'))
IDS.add(MaxPooling1D(pool_size=2))
IDS.add(LSTM(70, dropout=0.1))
IDS.add(Dropout(0.1))
IDS.add(Dense(2))
IDS.add(Activation('softmax'))
IDS.load_weights("models/ids/weight.h5")

result=IDS.predict(test_data)
print(result[0])

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(result.argmax(axis=1)):
    temp_result = result.argmax(axis=1)[i]
    temp_test = data.test.labels[i]
    if temp_result == temp_test:
        if temp_result == 0:
            TP = TP + 1
        else:
            TN = TN + 1
    else:
        if temp_result == 1:
            FP = FP + 1
        else:
            FN = FN + 1

print('Accuracy: ', (TP+TN)/(TP+TN+FP+FN))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', 2*(precision*recall)/(precision+recall))


G=torch.load('G_model.pth')
z = Variable(torch.randn(64,9))
G_sample = G(z)
G_sample = G_sample.detach().numpy()
sample_data = G_sample.reshape(G_sample.shape[0], G_sample.shape[1], 1)
I_sample=IDS.predict(sample_data).argmax(axis=1)
# I_sample=IDS.predict(G_sample)
print(I_sample)
