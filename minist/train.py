import os.path
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块
import matplotlib.pyplot as plt
from matplotlib import cm
from model import CNN


def main():
    torch.manual_seed(1)  # reproducible
    # Hyper Parameters
    EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
    BATCH_SIZE = 50
    LR = 0.001  # 学习率

    if not os.path.exists("./dataset") or not os.listdir("./dataset/mnist"):
        os.mkdir("./dataset")
        DOWNLOAD_MNIST = True
    else:
        DOWNLOAD_MNIST = False
    # Mnist 手写数字
    train_data = torchvision.datasets.MNIST(
        root="./dataset/mnist/",  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
    )

    print(f"train data size:{train_data.train_data.size()}")
    print(f"train labels size:{train_data.train_labels.size()}")
    # 绘制其中一个训练集数据
    # plt.imshow(train_data.train_data[0].numpy(), cmap="gray")
    # plt.title(f"labels: {train_data.train_labels[0]}")
    # plt.show()

    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 为了节约时间, 我们测试时只测试前2000个
    test_data = torchvision.datasets.MNIST(root="./dataset/mnist/", train=False)
    test_x = (
        torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]
        / 255.0
    )  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]

    cnn = CNN()
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    try:
        from sklearn.manifold import TSNE
        HAS_SK = True
    except:
        HAS_SK = False
        print("Please install sklearn for layer visualization")

    def plot_with_labels(lowDWeights, labels):
        plt.cla()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 * s / 9))
            plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.title("Visualize last layer")
        plt.show()
        plt.pause(0.01)

    plt.ion()
    # training and testing
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            output = cnn(b_x)[0]  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            # print(f"⚙️epoch{epoch + 1} / {EPOCH} | step {step}")

            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print(f"Epoch: {epoch + 1} / {EPOCH} | step: {step} | train loss: {loss.data.numpy(): .4f} | test accuracy: {accuracy: .2f}")
                if HAS_SK:
                    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)

    plt.ioff()

    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, "prediction number")
    print(test_y[:10].numpy(), "real number")


if __name__ == "__main__":
    main()
