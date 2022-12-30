import torch.nn as nn


def _set_init(layer):     # 参数初始化
    init.normal_(layer.weight, mean=0., std=.1)
    init.constant_(layer.bias, B_INIT)


class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []   # 太多层了, 我们用 for loop 建立
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)   # 给 input 的 BN

        for i in range(N_HIDDEN):               # 建层
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % i, fc)       # 注意! pytorch 一定要你将层信息变成 class 的属性! 我在这里花了2天时间发现了这个 bug
            _set_init(fc)                  # 参数初始化
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)   # 注意! pytorch 一定要你将层信息变成 class 的属性! 我在这里花了2天时间发现了这个 bug
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)         # output layer
        _set_init(self.predict)            # 参数初始化

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)    # 判断是否要加 BN
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)    # 为之后出图
            if self.do_bn: x = self.bns[i](x)  # 判断是否要加 BN
            x = ACTIVATION(x)
            layer_input.append(x)       # 为之后出图
        out = self.predict(x)
        return out, layer_input, pre_activation

# 建立两个 net, 一个有 BN, 一个没有
nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
