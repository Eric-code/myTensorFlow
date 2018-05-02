import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 2000               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1000
TIME_STEP = 1          # rnn time step / image height
INPUT_SIZE = 19        # rnn input size / image width
LR = 0.001               # learning rate


class feature_dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.data_list = []
        self.label_list = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = list(line.split(','))
                # line.pop(-1)
                line = [float(x) for x in line]
                # print(line[-1])
                self.data_list.append(line[:-1])
                self.label_list.append(line[-1])

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        return data, label

    def __len__(self):
        return len(self.label_list)


train_data = feature_dataset('flow_15s_all_19_train.csv')
train_data.data_list = np.array(train_data.data_list)
train_data.data_list = train_data.data_list.reshape((-1, TIME_STEP, INPUT_SIZE))  # (15642, 5, 19)
train_x = torch.from_numpy(train_data.data_list)
train_x = Variable(train_x).type(torch.FloatTensor)
train_y = np.array(train_data.label_list).squeeze()
# train_data.label_list = np.array(train_data.label_list)
test_data = feature_dataset('flow_15s_all_19_test.csv')
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
a = np.array(test_data.data_list)
a = a.reshape((-1, TIME_STEP, INPUT_SIZE))
test_feature = torch.from_numpy(a)
test_x = Variable(test_feature).type(torch.FloatTensor)
test_y = np.array(test_data.label_list).squeeze()
b = np.array(test_data.label_list)
b = torch.from_numpy(b)
# test_y = Variable(b).type(torch.LongTensor)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=1280,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # dropout=0.5,
        )

        self.out = nn.Linear(1280, 8)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
if torch.cuda.is_available():
    rnn = rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        x = x.float()
        y = y.long()
        b_x = Variable(x.view(-1, 1, 19))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                               # batch y

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        # print(train_accuracy)

        if step % 5 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            # print(pred_y.shape)
            # print(test_y.shape)
            # print(type(pred_y))
            # print(type(test_y))
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            # output2 = rnn(train_x)
            # pred = torch.max(output2, 1)[1].data.numpy().squeeze()
            # train_accuracy = sum(pred == train_y) / float(train_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.item(), '| test accuracy: %.2f' % accuracy)

