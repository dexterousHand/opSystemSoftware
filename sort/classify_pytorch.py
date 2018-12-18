import numpy as np
import torch
import torch.nn as nn

# 数据处理完成，卷积训练


# Hyper Parameters
EPOCH = 50  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 60
LR = 0.001  # learning rate
save_model = False


def load_test_data(filename):
    test_data, test_label = quickload(filename)
    test_data = test_data[:, np.newaxis]

    test_data = torch.tensor(test_data).cuda()
    test_label = torch.tensor(test_label).cuda()
    return test_data, test_label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 120, 160)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 60, 80)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 60, 80)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(2),  # output shape (32, 30, 40)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 30, 40)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (64, 30, 40)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(2),  # output shape (64, 15, 20)
        )
        self.conv4 = nn.Sequential(  # input shape (64, 15, 20)
            nn.Conv2d(64, 128, 5, 1, 2),  # output shape (128, 15, 20)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activation
        )
        self.conv5 = nn.Sequential(  # input shape (128, 15, 20)
            nn.Conv2d(128, 256, 5, 1, 2),  # output shape (128, 15, 20)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # activation
        )
        self.out = nn.Sequential(
            nn.Linear(256 * 15 * 20, 1024),
            nn.Dropout(0.7),
            nn.Linear(1024, 3)
        )  # fully connected layer, output 3 classes

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x.float())
        x = self.conv3(x.float())
        x = self.conv4(x.float())
        x = self.conv5(x.float())
        # for i in range(3):
        #     x=self.conv(x.float())
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


def restore_params(model):
    print("restore parameters")
    model.conv1.load_state_dict(torch.load('./sort/conv1.pkl'))
    model.conv2.load_state_dict(torch.load('./sort/conv2.pkl'))
    model.conv3.load_state_dict(torch.load('./sort/conv3.pkl'))
    model.conv4.load_state_dict(torch.load('./sort/conv4.pkl'))
    model.conv5.load_state_dict(torch.load('./sort/conv5.pkl'))
    model.out.load_state_dict(torch.load('./sort/out.pkl'))


def test(model, test_data, test_label):
    test_output, _ = model(test_data)
    test_output = test_output.cuda()
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
    accuracy = float((pred_y == test_label.cpu().data.numpy()).astype(int).sum()) / float(test_label.size(0))
    return accuracy


def loadimg(path):
    print("load img:", path)
    import cv2
    img = cv2.imread(path, 0)
    for i in range(2):
        img = cv2.pyrDown(img)
    img = ((img.reshape(-1) - 128) / 128).reshape(img.shape)[np.newaxis, np.newaxis]
    return img


def test_sample(model, data, label):
    # data=data
    test_output, _ = model(data)
    test_output = test_output.cuda()
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()[0]
    print("predict: ", pred_y)
    print("real   : ", int(label))
    return str(pred_y)


def quickload(filename):
    print("quick load method")
    data = np.load(filename)
    # normalization
    data = ((data.reshape(-1) - 128) / 128).reshape(data.shape)
    rawData = []
    (_, n, l, w) = data.shape
    for i in range(3):
        # 给数据加上标注信息
        label = (np.ones(n) * (i))[:, np.newaxis]
        # 图片矩阵拉平
        rawData.append(np.concatenate((data[i].reshape(n, -1), label), axis=1))
    # 合并3类数据并shuffle
    datus = np.array(rawData).reshape(3 * n, -1)
    # np.random.shuffle(datus)
    data = datus[:, :-1]
    label = datus[:, -1]
    data = data.reshape(-1, l, w)
    return data, label


print("model define")
cnn = CNN()
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# 恢复参数
restore_params(cnn)
test_data, test_label = load_test_data("./sort/test.npy")
# accuracy = test(cnn, test_data, test_label)
# print("test accuracy:",accuracy)

def guitest(index):
    sample_data = test_data[index].cpu().numpy()[np.newaxis]
    sample_label = test_label[index].cpu().numpy()[np.newaxis]
    # sample_data = loadimg("./sort/0")
    # sample_label = np.array(1)
    sample_data = torch.tensor(sample_data).cuda()
    pred=test_sample(cnn, sample_data, sample_label)
    return pred
