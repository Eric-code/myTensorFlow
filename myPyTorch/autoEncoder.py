import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import datetime


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


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
test_data = feature_dataset('flow_15s_all_19_test.csv')
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

train_data.data_list = np.array(train_data.data_list)
train_data.data_list = torch.from_numpy(train_data.data_list)
train_data.data_list = Variable(train_data.data_list).type(torch.FloatTensor)
train_data.label_list = np.array(train_data.label_list)
train_data.label_list = torch.from_numpy(train_data.label_list)
train_data.label_list = Variable(train_data.label_list).type(torch.FloatTensor)
# plot one example
print(train_data.data_list.size())     # (60000, 28, 28)
print(train_data.label_list.size())   # (60000)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1*19, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.Linear(12, 8),
            nn.Tanh(),
            nn.Linear(8, 4),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 12),
            nn.Tanh(),
            nn.Linear(12, 16),
            nn.Tanh(),
            nn.Linear(16, 1*19),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
# f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
# plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = Variable(train_data.data_list[:N_TEST_IMG].view(-1, 1*19).type(torch.FloatTensor))
# for i in range(N_TEST_IMG):
#     a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.float()
        y = y.long()
        b_x = Variable(x.view(-1, 1*19))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 1*19))   # batch y, shape (batch, 28*28)
        b_label = Variable(y)               # batch label

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)


# plt.ioff()
# plt.show()

# visualize in 3D plot
view_data = Variable(train_data.data_list[:].view(-1, 1*19).type(torch.FloatTensor))
values = train_data.label_list[:].numpy()
print(values.shape)
encoded_data, _ = autoencoder(view_data)
# print(encoded_data)
encoded_train = encoded_data.detach().numpy()
print(encoded_train.shape)
encoded_train = encoded_train.tolist()
x_train, x_test, y_train, y_test = train_test_split(encoded_train, values, test_size=1/5., random_state=8)  # 分割训练集和测试集
estimators = {}
estimators['tree'] = tree.DecisionTreeClassifier(criterion='gini', random_state=8)  # 决策树
estimators['forest'] = RandomForestClassifier(n_estimators=20, criterion='gini', bootstrap=True, n_jobs=2, random_state=8)  # 随机森林
for k in estimators.keys():
    start_time = datetime.datetime.now()
    print('----%s----' % k)
    estimators[k] = estimators[k].fit(x_train, y_train)
    pred = estimators[k].predict(x_test)
    print(pred[:10])
    print("%s Score: %0.4f" % (k, estimators[k].score(x_test, y_test)))
    scores = cross_val_score(estimators[k], x_train, y_train, scoring='accuracy', cv=10)
    print("%s Cross Avg. Score: %0.4f (+/- %0.4f)" % (k, scores.mean(), scores.std() * 2))
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print("%s Time: %0.2f" % (k, time_spend.total_seconds()))
# fig = plt.figure()
# ax = Axes3D(fig)
# X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
# values = train_data.label_list[:100:20000].numpy()
# for x, y, z, s in zip(X, Y, Z, values):
#     c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
# ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
# plt.show()

