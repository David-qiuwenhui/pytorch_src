def main():
    import torch
    import torch.utils.data as Data
    torch.manual_seed(1)  # reproducible

    BATCH_SIZE = 8  # 批训练的数据个数
    # BATCH_SIZE = 5

    x = torch.linspace(1, 10, 10)  # x data (torch tensor)
    y = torch.linspace(10, 1, 10)  # y data (torch tensor)

    # 先转换成 torch 能识别的 Dataset
    torch_dataset = Data.TensorDataset(x, y)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )

    # 训练所有整套数据 3 次
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            # 假设这里就是你训练的地方...

            # 打出来一些数据
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

    """
    BATCH_SIZE = 5
    Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
    Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
    Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
    Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
    Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
    Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
    
    BATCH_SIZE = 8
    Epoch:  0 | Step:  0 | batch x:  [ 5.  3.  1.  7.  9.  8. 10.  2.] | batch y:  [ 6.  8. 10.  4.  2.  3.  1.  9.]
    Epoch:  0 | Step:  1 | batch x:  [6. 4.] | batch y:  [5. 7.]
    Epoch:  1 | Step:  0 | batch x:  [5. 9. 2. 6. 1. 3. 4. 7.] | batch y:  [ 6.  2.  9.  5. 10.  8.  7.  4.]
    Epoch:  1 | Step:  1 | batch x:  [10.  8.] | batch y:  [1. 3.]
    Epoch:  2 | Step:  0 | batch x:  [ 6.  9.  4.  8.  7. 10.  3.  2.] | batch y:  [5. 2. 7. 3. 4. 1. 8. 9.]
    Epoch:  2 | Step:  1 | batch x:  [1. 5.] | batch y:  [10.  6.]
    """


if __name__ == "__main__":
    main()