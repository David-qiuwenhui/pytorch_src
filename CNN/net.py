def main():
    import torch
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net1 = Net(1, 10, 1)  # 这是我们用这种方式搭建的 net1
    print(net1)
    # Net(
    #   (hidden): Linear(in_features=1, out_features=10, bias=True)
    #   (predict): Linear(in_features=10, out_features=1, bias=True)
    # )

    net2 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    print(net2)
    # Sequential(
    #   (0): Linear(in_features=1, out_features=10, bias=True)
    #   (1): ReLU()
    #   (2): Linear(in_features=10, out_features=1, bias=True)
    # )


if __name__ == '__main__':
    main()