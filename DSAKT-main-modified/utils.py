import math

import torch
import random


# 获取数据
def getdata(window_size, path, model_type, drop=False):
    '''
    @param model_type: 'sakt' or 'saint'
    '''
    N = 0  # N表示input列表中有多少行数据
    count = 0  # 计数器,用来判断当前是哪一行
    # 每行数据最多有多少个
    E = -1
    units = []
    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    bis = 0
    # 读取路径
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if count % 3 == 0:
            pass
        # 第二行，题目编号
        elif count % 3 == 1:
            # 换行符拆分了那么[0]就是数据，在根据,拆分就是每个问题编号 tlist [1,10,25]
            tlst = line.split('\n')[0].split(',')
            # 找到最大题目编号并存入E
            for item in tlst:
                if int(item) > E:
                    E = int(item)

            tlst_1 = tlst[0:len(tlst) - 1]  # 取第一个到倒数第二个
            tlst_2 = tlst[1:len(tlst)]  # 取第二个到最后一个

            # drop默认false
            '''if drop:
                if len(tlst_1) > window_size:
                    tlst_1 = tlst_1[0:window_size]
                    tlst_2 = tlst_2[0:window_size]'''
            # 根据window_size进行切片处理
            while len(tlst_1) > window_size:
                input_1.append([int(i) + 1 for i in tlst_1[0:window_size]])
                # 每次切片 就加1
                N += 1
                tlst_1 = tlst_1[window_size:len(tlst_1)]
                units.append(window_size)
            # units记录每段数据的实际大小
            units.append(len(tlst_1))
            # 结果的列数就是window_size，给不足window_size的补0   padding mask
            tlst_1 = [int(i) + 1 for i in tlst_1] + [0] * (window_size - len(tlst_1))
            N += 1
            input_1.append(tlst_1)

            # 对tlst2进行切片变成 [[10,20,30],[20,505,20]]
            while len(tlst_2) > window_size:
                input_3.append([int(i) + 1 for i in tlst_2[0:window_size]])
                tlst_2 = tlst_2[window_size:len(tlst_2)]
            # 结果的列数就是window_size
            tlst_2 = [int(i) + 1 for i in tlst_2] + [0] * (window_size - len(tlst_2))
            input_3.append(tlst_2)
        # 第三行 学生做题结果 0/1
        else:  # 1:False 2:True
            tlst = line.split('\n')[0].split(',')
            tlst_1 = tlst[0:len(tlst) - 1]
            tlst_2 = tlst[1:len(tlst)]

            '''if drop:
                if len(tlst_1) > window_size:
                    tlst_1 = tlst_1[0:window_size]
                    tlst_2 = tlst_2[0:window_size]'''

            while len(tlst_1) > window_size:
                input_2.append([int(i) + bis for i in tlst_1[0:window_size]])
                tlst_1 = tlst_1[window_size:len(tlst_1)]
            tlst_1 = [int(i) + bis for i in tlst_1] + [0] * (window_size - len(tlst_1))
            input_2.append(tlst_1)

            while len(tlst_2) > window_size:
                input_4.append([int(i) + bis for i in tlst_2[0:window_size]])
                tlst_2 = tlst_2[window_size:len(tlst_2)]
            tlst_2 = [int(i) + bis for i in tlst_2] + [0] * (window_size - len(tlst_2))
            input_4.append(tlst_2)
        # 下三行数据
        count += 1
    # 读取完所有数据，关闭文件
    file.close()
    # 最大题目编号加1
    E += 1  # 26688

    # torch.Size([40677, 100])
    input_1 = torch.tensor(input_1)
    # torch.Size([40677, 100])
    input_2 = torch.tensor(input_2)
    # torch.Size([40677, 100])
    input_3 = torch.tensor(input_3)
    # torch.Size([40677, 100])
    input_4 = torch.tensor(input_4)
    if model_type == 'sakt':
        # tensor做+号运算，维度相同的情况下，结果维度也一样
        input_1 = input_1 + E * input_2
        # torch.stack 就是将几个矩阵拼接起来，形成更大的矩阵 变成torch.Size([3,40677, 100])
        # N是40677，就是input1的行数  E26688
        # units 如果某行超过50【window_size】个数，则记录50，剩下的依次再分段记录，若不超过50，则记录实际数量，如第一行记录5
        return torch.stack((input_1, input_3, input_4), 0), N, E, units
    elif model_type == 'saint':
        return torch.stack((input_1, input_2), 0), N, E, units
    else:
        raise Exception('model type error')



# 加载数据
def dataloader(data, batch_size, shuffle: bool):
    # 转换原有的维度 本来是3 40677 100  变成40677 3 100
    data = data.permute(1, 0, 2)
    lis = [x for x in range(len(data))]  # lis就是[0,1,...40676]
    if shuffle:
        random.shuffle(lis)  # 随机打乱比如[250,36,0,....152]
    # 生成单精度浮点类型的张量 再使用long变为长整型
    # 这个和torch.tensor(lis)有什么区别呢
    lis = torch.Tensor(lis).long()
    ret = []
    for i in range(int(len(data) / batch_size)):  # 317
        # index_select从data中按照dim=0，也就是第一维40677的数字按照lis返回的行数进行返回给temp
        # 如lis里面有[123,215..]，那么就取data的123行和215行这个数据给temp，
        # temp的tensor.size就是[128,3,100]
        temp = torch.index_select(data, 0, lis[i * batch_size: (i + 1) * batch_size])  # 每个temp是(128,3,100)
        # ret(317,128,3,100)
        ret.append(temp)
    # 这里使用permute的原因是 在开头就对data使用了permute，为了将数据打乱，经过打乱后
    # 需要回到原来的shape，又需要执行permute
    # 变成(317,3,128,100) stack是为了让ret变成tensor
    return torch.stack(ret, 0).permute(0, 2, 1, 3)


class NoamOpt:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, dimension: int, factor=0.1):
        self.optimizer = optimizer
        self._steps = 0
        self._warmup = warmup
        self._factor = factor
        self._dimension = dimension

    def step(self):
        self._steps += 1
        rate = self._factor * (
                self._dimension ** (-0.5) * min(self._steps ** (-0.5), self._steps * self._warmup ** (-1.5)))
        # rate = self._factor * (
        #        math.pow(self._dimension, -0.5) * min(math.pow(self._steps, -0.5),
        #                                              math.pow(self._steps * self._warmup, -1.5))
        # )
        # print("第{}步的learning rate是{}".format(self._steps, rate))
        for x in self.optimizer.param_groups:
            x['lr'] = rate
