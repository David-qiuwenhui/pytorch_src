import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 超参数
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = F.tanh  # 你可以换 relu 试试
B_INIT = -0.2  # 模拟不好的 参数初始化

# training data
x = np.linspace(-7, 10, num=N_SAMPLES)[:, np.newaxis]
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise

# test data
test_x = np.linspace(-7, 10, num=200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + noise

train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

# show data
plt.scatter(
    x=train_x.numpy(), y=train_y.numpy(), c="#FF9359", s=50, alpha=0.2, label="train"
)
plt.legend(loc="upper left")


# plt.show()


class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []  # 太多层了, 我们用 for loop 建立
        self.bns = []
        self.bn_input = nn.BatchNorm1d(num_features=1, momentum=0.5)  # 给 input 的 BN

        for i in range(N_HIDDEN):  # 建层
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(in_features=input_size, out_features=10)
            setattr(
                self, "fc%i" % i, fc
            )  # 注意! pytorch 一定要你将层信息变成 class 的属性! 我在这里花了2天时间发现了这个 bug
            self._set_init(fc)  # 参数初始化
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(num_features=10, momentum=0.5)
                setattr(
                    self, "bn%i" % i, bn
                )  # 注意! pytorch 一定要你将层信息变成 class 的属性! 我在这里花了2天时间发现了这个 bug
                self.bns.append(bn)

        self.predict = nn.Linear(in_features=10, out_features=1)  # output layer
        self._set_init(self.predict)  # 参数初始化

    def _set_init(self, layer):  # 参数初始化
        init.normal_(layer.weight, mean=0.0, std=0.1)
        init.constant_(layer.bias, val=B_INIT)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn:
            x = self.bn_input(x)  # 判断是否要加 BN
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)  # 为之后出图
            if self.do_bn:
                x = self.bns[i](x)  # 判断是否要加 BN
            x = ACTIVATION(x)
            layer_input.append(x)  # 为之后出图
        out = self.predict(x)
        return out, layer_input, pre_activation


# 建立两个 net, 一个有 BN, 一个没有
nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
"""
Net(
    (bn_input): BatchNorm1d(1, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc0): Linear(in_features=1, out_features=10, bias=True)
    (fc1): Linear(in_features=10, out_features=10, bias=True)
    (fc2): Linear(in_features=10, out_features=10, bias=True)
    (fc3): Linear(in_features=10, out_features=10, bias=True)
    (fc4): Linear(in_features=10, out_features=10, bias=True)
    (fc5): Linear(in_features=10, out_features=10, bias=True)
    (fc6): Linear(in_features=10, out_features=10, bias=True)
    (fc7): Linear(in_features=10, out_features=10, bias=True)
    (predict): Linear(in_features=10, out_features=1, bias=True)
)

Net(
    (bn_input): BatchNorm1d(1, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc0): Linear(in_features=1, out_features=10, bias=True)
    (bn0): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc1): Linear(in_features=10, out_features=10, bias=True)
    (bn1): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=10, out_features=10, bias=True)
    (bn2): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc3): Linear(in_features=10, out_features=10, bias=True)
    (bn3): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc4): Linear(in_features=10, out_features=10, bias=True)
    (bn4): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc5): Linear(in_features=10, out_features=10, bias=True)
    (bn5): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc6): Linear(in_features=10, out_features=10, bias=True)
    (bn6): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (fc7): Linear(in_features=10, out_features=10, bias=True)
    (bn7): BatchNorm1d(10, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True)
    (predict): Linear(in_features=10, out_features=1, bias=True)
)
"""

opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
loss_func = torch.nn.MSELoss()


def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(
        zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])
    ):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)
        ax_pa.set_title("L" + str(i))
        ax_pa.hist(
            pre_ac[i].data.numpy().ravel(),
            bins=10,
            range=p_range,
            color="#FF9359",
            alpha=0.5,
        )
        ax_pa_bn.hist(
            pre_ac_bn[i].data.numpy().ravel(),
            bins=10,
            range=p_range,
            color="#74BCFF",
            alpha=0.5,
        )
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color="#FF9359")
        ax_bn.hist(
            l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color="#74BCFF"
        )
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(())
            a.set_xticks(())
        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel("PreAct")
        axs[1, 0].set_ylabel("BN PreAct")
        axs[2, 0].set_ylabel("Act")
        axs[3, 0].set_ylabel("BN Act")
    plt.pause(0.01)


if __name__ == "__main__":
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()  # something about plotting
    plt.show()

    # training
    losses = [[], []]  # 每个网络一个 list 来记录误差
    for epoch in range(EPOCH):
        print("Epoch: ", epoch)
        layer_inputs, pre_acts = [], []

        for net, l in zip(nets, losses):
            net.eval()
            pred, layer_input, pre_act = net(test_x)
            l.append(loss_func(pred, test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()
        plot_histogram(*layer_inputs, *pre_acts)

        for step, (b_x, b_y) in enumerate(train_loader):
            for net, opt in zip(nets, opts):  # 训练两个网络
                pred, _, _ = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()  # 这也会训练 BN 里面的参数

    plt.ioff()

    # plot training loss
    plt.figure(2)
    plt.plot(losses[0], c="#FF9359", lw=3, label="Original")
    plt.plot(losses[1], c="#74BCFF", lw=3, label="Batch Normalization")
    plt.xlabel("step")
    plt.ylabel("test loss")
    plt.ylim((0, 2000))
    plt.legend(loc="best")

    # evaluation
    # set net to eval mode to freeze the parameters in batch normalization layers
    [net.eval() for net in nets]  # set eval mode to fix moving_mean and moving_var
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(
        test_x.data.numpy(), preds[0].data.numpy(), c="#FF9359", lw=4, label="Original"
    )
    plt.plot(
        test_x.data.numpy(),
        preds[1].data.numpy(),
        c="#74BCFF",
        lw=4,
        label="Batch Normalization",
    )
    plt.scatter(
        test_x.data.numpy(), test_y.data.numpy(), c="r", s=50, alpha=0.2, label="train"
    )
    plt.legend(loc="best")
    plt.show()
