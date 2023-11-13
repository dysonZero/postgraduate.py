import math
import queue
import sys

import torch


def test01():
    import torch
    print(torch.cuda.is_available())

    l1 = [[10, 20], [40, 30], [60, 30]]
    l2 = [[10, 20], [40, 30], [60, 30]]
    t1 = torch.tensor(l1)
    print(t1)  # 3*2
    t2 = torch.tensor(l2)
    print(t2)  # 3*2
    data = torch.stack((t1, t2), 0)
    print(data)
    data = data.permute(1, 0, 2)
    print(data, type(data))
    t3 = torch.tensor([1, 2, 15, 3]).long()
    print(t3)


def test02():
    # toTensor
    from PIL import Image
    from torchvision import transforms
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter("..//logs")
    img = Image.open("..//dsakt.png").convert('RGB')
    print(img)

    trans_totensor = transforms.ToTensor()
    img_tensor = trans_totensor(img)
    writer.add_image("ToTensor", img_tensor)
    writer.close()

    # Normalize
    trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_norm = trans_norm(img_tensor)
    print(img_norm[0][0][0])


class Person:
    def __call__(self, name):
        print("__call__" + "Hello" + name)

    def hello(self, name):
        print("hello" + name)


def test03():
    l1 = [2, 3, 4, 5, 6]
    l2 = [x for x in l1] + [0] * 10
    print(l2)


def test04(s):
    s = s.strip(' ')
    ret_str = ''
    i = j = 0
    while j < len(s):
        # 当j到达最后一个位置时 直接加
        if j == len(s) - 1 or (s[j + 1] == ' ' and s[j] != ' '):
            ret_str = s[i:j + 1] + ' ' + ret_str
            j = j + 1
        elif s[j] == ' ' and s[j + 1] != ' ':
            j = j + 1
            i = j
        else:
            j = j + 1
    return ret_str.strip(' ')


def test05(num):
    s = 0
    while num > 0:
        s += num % 10
        num //= 10
    return s


def test06():
    import torch
    ten1 = torch.tensor([[[10, 20, 30], [10, 10, 10]], [[20, 10, 7], [10, 9, 8]]])
    print(ten1)
    ten1 = ten1.permute(1, 0, 2)
    print(ten1)
    print(torch.cuda.get_device_name())


def test07():
    import torch
    # 生成0-11的张量
    x = torch.arange(12)
    print(x)
    print(x.shape)
    # 元素总数
    print(x.numel())
    # 重新换形状
    print(x.reshape(3, 4))
    print(torch.zeros((2, 3, 4)))
    print(torch.tensor([10, 20, 30]))
    print(torch.tensor([[10, 20, 30], [10, 10, 10]]))

    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x == y)
    print(x - y, x + y, x * y, x / y)

    x = torch.tensor([[10, 20, 30], [10, 10, 10]])
    print(torch.reshape(x, (3, 2)))
    y = torch.tensor([[1, 2, 3], [1, 1, 1]])

    print(torch.cat((x, y), dim=0))
    print(torch.cat((x, y), dim=1))

    a = torch.ones(2, 5, 4)
    print(a)
    print(a.sum(dim=1))
    print(a.sum(dim=0, keepdim=True))
    print(a.sum(dim=2, keepdim=True))
    print(a.sum(dim=[0, 1], keepdim=True))


# 测试自动求导
def test08():
    import torch
    x = torch.arange(4.0)
    print(x)
    x.requires_grad_(True)
    print(x.grad)  # 默认是None

    y = 2 * torch.dot(x, x)
    print(y)

    y.backward()
    print(x.grad)
    print(x.grad == 4 * x)

    # pytorch会累积梯度，需要清除之前的值
    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(x.grad)

    x.grad.zero_()
    y = x * x
    y.sum().backward()
    print(x.grad)

    x.grad.zero_()
    y = x * x
    u = y.detach()  # 需要detach才能转
    z = u * x
    # 计算梯度是一个耗资源的事情，所以需要显式申明
    z.sum().backward()
    print(x.grad == u)


def test09():
    print(60 ^ 13)
    print((60 & 13) << 1)
    print(0x7fffffff)
    print(bin(4294967295))
    print(0 << 2)


def test10():
    for i in range(3 - 1, 0 - 1, -1):
        print(i)
    print(math.pow(2, 31) - 1)
    print(int('132'))
    print()
    import queue as q
    q = queue.Queue()
    q.put(1)
    q.put(2)
    print(q.get())

    print(q.get())
    print(not q)
    print(q.empty())
    print([i for i in range(10)])
    from torch.utils.data import Dataset


if __name__ == "__main__":
    # test10()
    '''import time
    print(time.localtime())
    print(time.strftime("%Y%m%d%H%M", time.localtime()))
    import torch'''
    # print(math.pow(64,-0.5))
    import time
    import pandas as pd

    a = torch.tensor([[[6],[5]],[[4],[3]],[[2],[1]]])
    b = torch.tensor([[[1,2,3],[4,5,6]]])
    print(a.shape)
    print(b.shape)
    print(a*b)
