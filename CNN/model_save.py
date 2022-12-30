import matplotlib.pyplot as plt


def main():
    import torch

    # 我们快速地建造数据, 搭建网络
    torch.manual_seed(1)  # reproducible
    # 假数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    def save():
        # net1
        net1 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
        optimizer = torch.optim.SGD(params=net1.parameters(), lr=0.5)
        loss_func = torch.nn.MSELoss()

        # 训练
        epochs = 100
        for t in range(epochs):
            prediction = net1(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # plot result
        plt.figure(1, figsize=(10, 3))
        plt.subplot(131)
        plt.title('Net1')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

        # two ways to save the net
        torch.save(net1, 'net.pkl')  # 保存整个模型
        torch.save(net1.state_dict(), 'net_params.pkl')  # 仅保存模型的参数

    # net2读取整个模型
    def restore_net():
        # restore entire net1 to net2
        net2 = torch.load('net.pkl')
        prediction = net2(x)

        # plot result
        plt.subplot(132)
        plt.title('Net2')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # net3 只读取模型的参数
    def restore_params():
        net3 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

        # 将保存的参数复制到 net3
        net3.load_state_dict(torch.load('net_params.pkl'))
        prediction = net3(x)

        # plot result
        plt.subplot(133)
        plt.title('Net3')
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

        plt.tight_layout()
        plt.show()

    # 第三步 显示结果
    # 保存 net1 (1. 整个模型, 2. 只有模型参数)
    save()

    # 提取整个网络
    restore_net()

    # 提取网络参数, 复制到新网络
    restore_params()


if __name__ == '__main__':
    main()
