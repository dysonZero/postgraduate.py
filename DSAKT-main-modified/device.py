import torch


def return_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("输出的计算引擎：",device)
    # device = torch.device('cpu')
    return device
