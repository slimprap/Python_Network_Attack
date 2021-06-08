#We shall use test set here. Generator G will take test sample as input and give fake malicious sample as output.
#IDS would try to classify this fake sample
import transform
import torch
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import Conv1D, MaxPooling1D
from torch.autograd import Variable

data = transform.data_importer_IDS(Evaluate=False)
train_data= data.train.samples.values.reshape(data.train.samples.shape[0], data.train.samples.shape[1], 1)
test_data= data.test.samples.values.reshape(data.test.samples.shape[0], data.test.samples.shape[1], 1)

data_partitioned = transform.data_importer_IDS(Evaluate=True)
test_data_attack= data_partitioned.validation.samples.values.reshape(data_partitioned.validation.samples.shape[0], data_partitioned.validation.samples.shape[1], 1)
test_data_normal= data_partitioned.test.samples.values.reshape(data_partitioned.test.samples.shape[0], data_partitioned.test.samples.shape[1], 1)
# print('normal data attack cat', data.test.samples[0]['attack_cat'])
IDS=Sequential()
IDS.add(Conv1D(32, 3, input_shape=train_data.shape[1:]))
IDS.add(Activation('relu'))
IDS.add(Conv1D(64, 3))
IDS.add(Activation('relu'))
IDS.add(MaxPooling1D(pool_size=2))
IDS.add(LSTM(70, dropout=0.1))
IDS.add(Dropout(0.1))
IDS.add(Dense(10))
IDS.add(Activation('softmax'))
IDS.load_weights("models/ids/weight.h5")

result=IDS.predict(test_data)
result_normal=IDS.predict(test_data_normal)
result_attack=IDS.predict(test_data_attack)
# result=IDS.predict(train_data)
# print("testLabel before processing:", data.test.labels[0:5])

TP = TP_NORMAL = TP_ATTACK = 0
TN = TN_NORMAL = TN_ATTACK = 0
FP = FP_NORMAL = FP_ATTACK = 0
FN = FN_NORMAL = FN_ATTACK = 0

classResult=result.argmax(axis=1)
classResultNormal=result_normal.argmax(axis=1)
classResultAttack=result_attack.argmax(axis=1)
# print("classResult before processing:",result[0:5])
# classResult =(result>0.5)
testLabels=data.test.labels.argmax(axis=1)
testLabelsNormal=data_partitioned.test.labels.argmax(axis=1)
testLabelsAttack=data_partitioned.validation.labels.argmax(axis=1)
# testLabels=data.train.labels.argmax(axis=1)
# print("testLabels", testLabels[0:5])
# print("classResult",classResult[0:5])
numTestSample=classResult.shape[0]
print("total test sample", numTestSample)
for i in range(numTestSample):
    temp_result = classResult[i]
    temp_test = testLabels[i]
    if temp_result == temp_test:
        if temp_result == 1:
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

numTestSampleNormal=classResultNormal.shape[0]
print("test sample normal", numTestSampleNormal)
print("test sample attack", classResultAttack.shape[0])
for i in range(numTestSampleNormal):
    temp_result = classResultNormal[i]
    temp_test = testLabelsNormal[i]
    if temp_result == temp_test:
        if temp_result == 1:
            TP_NORMAL = TP_NORMAL + 1
        else:
            TN_NORMAL = TN_NORMAL + 1
    else:
        if temp_result == 1:
            FP_NORMAL = FP_NORMAL + 1
        else:
            FN_NORMAL = FN_NORMAL + 1

print('Accuracy Normal: ', (TP_NORMAL+TN_NORMAL)/(TP_NORMAL+TN_NORMAL+FP_NORMAL+FN_NORMAL))
# precision = TP_NORMAL/(TP_NORMAL+FP_NORMAL)
# recall = TP_NORMAL/(TP_NORMAL+FN_NORMAL)
# print('Precision Normal: ', precision)
# print('Recall Normal: ', recall)
# print('F1: ', 2*(precision*recall)/(precision+recall))

numTestSampleAttack=classResultAttack.shape[0]
# print("numTestSample", numTestSample)
for i in range(numTestSampleAttack):
    temp_result = classResultAttack[i]
    temp_test = testLabelsAttack[i]
    if temp_result == temp_test:
        if temp_result == 1:
            TP_ATTACK = TP_ATTACK + 1
        else:
            TN_ATTACK = TN_ATTACK + 1
    else:
        if temp_result == 1:
            FP_ATTACK = FP_ATTACK + 1
        else:
            FN_ATTACK = FN_ATTACK + 1

print('Accuracy Attack: ', (TP_ATTACK+TN_ATTACK)/(TP_ATTACK+TN_ATTACK+FP_ATTACK+FN_ATTACK))
# precision = TP_ATTACK/(TP_ATTACK+FP_ATTACK)
# recall = TP_ATTACK/(TP_ATTACK+FN_ATTACK)
# print('Precision Attack: ', precision)
# print('Recall Attack: ', recall)
# print('F1: ', 2*(precision*recall)/(precision+recall))

G=torch.load('G_model.pth')
z = Variable(torch.randn(512,9))
G_sample = G(z)
print("G_sample shape:", G_sample.shape)
G_sample = G_sample.detach().numpy()
sample_data = G_sample.reshape(G_sample.shape[0], G_sample.shape[1], 1)
I_sample=IDS.predict(sample_data).argmax(axis=1)
correct=I_sample[I_sample==1]
totalG=I_sample.shape[0]
correctG=correct.shape[0]
# I_sample=IDS.predict(G_sample)
print("Accuracy under GAN:", correctG/totalG)
