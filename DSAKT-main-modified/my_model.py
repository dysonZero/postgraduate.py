import torch.nn as nn
import numpy as np
from model1 import MultiHeadAttention
from device import return_device
import math
import torch
import torch.optim as optim
from sklearn import metrics
from utils import getdata, dataloader, NoamOpt

# 可以用torch.cuda.get_device_name(0)查看当前显卡的名称
device = return_device()


# softmax 对于一个长度为K的任意实数矢量，softmax可以将它压缩成为一个长度为K、取值在（0,1）区间的实数矢量，且矢量元素之和为1
# dataset就是数据集、有索引进行访问,告诉数据在什么地方
# dataLoader

# 把拿到的数据进行中间的一个表示
class Encoder(nn.Module):
    # dim64 heads8 dropout7 window_size100
    def __init__(self, dim: int, heads: int, dropout: float, window_size: int):
        self.window_size = window_size
        super().__init__()
        # embed_dim 模型的总尺寸 num_heads 多头数，即embed_dim会通过num_heads数进行均分,即每个头有embed_dim/num_heads个维度
        self.MHA = MultiHeadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, window_size=window_size)
        # sequential 就是一个layer，相当于一层
        # 一个顺序容器，可以装进去很多模块，按顺序执行
        # Sequential 的 forward() 方法接受任何输入并将其转发到它包含的第一个模块
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.ReLU(),  # 返回比0大的数，小于0的直接等于0
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.Dropout(dropout))  # 防止过拟合
        # 将每个样本变成均值为1，方差为0
        # 对小批量输入应用层归一化
        self.LN = nn.LayerNorm(normalized_shape=dim)

    # data_in shape 128,100,64
    def forward(self, data_in):
        # data_out 128,100,64  Q K V
        data_out = self.MHA(data_in, data_in, data_in)
        # data_out 128 100 64
        data_out = self.LN(data_out + data_in)
        temp = data_out
        data_out = self.FFN(data_out)
        data_out = self.LN(data_out + temp)
        return data_out


class Decoder(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, window_size: int):
        self.window_size = window_size
        super(Decoder, self).__init__()
        self.MHA = MultiHeadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, window_size=window_size)
        self.FFN = nn.Sequential(
            # in_features 每个输入样本的大小  out_features每个输出样本的大小
            # 用于设置网络中的全连接层,如果输入的是二维张量，那么in_features就是[batch_size, size]的size
            # 全连接层的作用主要就是实现分类
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.Dropout(dropout))
        self.LN = nn.LayerNorm(normalized_shape=dim)

    def forward(self, data_in, encoded_data):
        data_out = self.MHA(data_in, data_in, data_in)
        data_out = self.LN(data_out + data_in)

        temp = data_out
        data_out = self.MHA(data_out, data_out, encoded_data)
        data_out = self.LN(data_out + temp)

        temp = data_out
        data_out = self.FFN(data_out)
        data_out = self.LN(data_out + temp)
        return data_out


# nn.Module其实就是一个类，这就是继承
# DSAKT就是自己的模型、必须继承nn.module
class My_Model(nn.Module):
    # 构造函数，当其他方法调用DASKT时，需要传递init里面的参数，并且进行初始化
    # init和super.init必须写，是模板 window_size100 dim64 heads8 dropout0.8
    def __init__(self, num_skills: int, window_size: int, dim: int, heads: int, dropout: float):
        super(My_Model, self).__init__()
        self.device = device
        self.window_size = window_size
        self.dim = dim
        self.loss_function = nn.BCELoss()
        self.Exerc_embedding = nn.Embedding(num_embeddings=2 * num_skills + 1, embedding_dim=dim, padding_idx=0)
        self.Query_embedding = nn.Embedding(num_embeddings=num_skills + 1, embedding_dim=dim, padding_idx=0)
        self.Projection = nn.ModuleList([nn.Linear(in_features=dim, out_features=dim, bias=False) for _ in range(2)])
        self.Prediction = nn.Sequential(nn.Linear(in_features=dim, out_features=1, bias=True), nn.Sigmoid())
        self.blstm = nn.LSTM(input_size=dim,
                             hidden_size=dim,
                             num_layers=1,
                             # dropout=dropout,
                             batch_first=True,
                             bidirectional=False)
        # 对位置进行编码
        temp = []
        # 公式的pos就是window_size,i就是维度
        for i in range(1, window_size + 1):
            # pos_t是一条震荡曲线
            pos_t = []
            for j in range(1, dim + 1):
                if j % 2 == 0:
                    pos_t.append(math.sin(i / math.pow(10000, j / dim)))
                else:
                    pos_t.append(math.cos(i / math.pow(10000, (j - 1) / dim)))
            temp.append(pos_t)  # temp的shape是[100,64]

        # unsqueeze 返回一个插入到指定位置的尺寸为 1 的新张量，如果设置为0，则是在外面再裹一层[]
        # posit_embeded shape是[1,100,64] 这更像是一个变量而非一个模型
        self.posit_embeded = torch.Tensor(temp).unsqueeze(0).to(self.device)
        # 实例化Encoder和decoder
        self.Encoder = Encoder(dim=dim, heads=heads, dropout=dropout, window_size=window_size)
        self.Decoder = Decoder(dim=dim, heads=heads, dropout=dropout, window_size=window_size)

    # forward就是神经网络运算的步骤  前向传播
    # ex_in,ex_qu的shape torch.Size([128, 100])
    def forward(self, ex_in, ex_qu):
        # 不需要计算梯度
        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
        # (ex_in != 0).unsqueeze(2)的shape  torch.Size([128, 100, 1])
        with torch.no_grad():
            # posi的shape torch.Size([128, 100, 64])
            # 这里的作用是针对ex_in不为0的部分进行位置编码
            posi = (ex_in != 0).unsqueeze(2) * self.posit_embeded
        # self.Exerc_embedding(ex_in)的shape  torch.Size([128, 100, 64])
        # interation shape 128 100 64
        interation = self.Exerc_embedding(ex_in)
        # question shape 128 100 64
        question = self.Query_embedding(ex_qu)

        interation, _ = self.blstm(interation)
        question, _ = self.blstm(question)
        # 线性变换 interation的shape 128 100 64 按照linear的算法，
        # 会按照里面的infeature 和outfeature构造一个权重矩阵A(outfeature,infeature)
        # 计算方法就是y = x*A.T+b(b是偏置)
        '''interation = self.Projection[0](interation)
        question = self.Projection[1](question)'''

        # 进入编码器架构
        # interation = self.Encoder(interation)
        # 将encoder的输出作为decoder的输入 进入解码器架构
        question = self.Decoder(question, interation)

        return self.Prediction(question)


def train_my_model(window_size: int, dim: int, heads: int, dropout: float, lr: float, train_path: str, valid_path: str,
                    batch_size: int, epochs: int):
    # 处理数据
    # train_data：tensor.size([3 42550 50]) unit_list_train没有使用
    train_data, N_train, E_train, unit_list_train = getdata(window_size=window_size, path=train_path, model_type='sakt')
    valid_data, N_val, E_test, unit_list_val = getdata(window_size=window_size, path=valid_path, model_type='sakt')
    # batch_size表示一次性运行多少个样本
    train_loader = dataloader(train_data, batch_size=batch_size, shuffle=True)
    E = max(E_train, E_test)  # 找出训练和测试最大的技能编号

    # 实例化模型
    model = My_Model(num_skills=E, window_size=window_size, dim=dim, heads=heads, dropout=dropout)
    # to方法 是nn.module的方法 将模型加载到相应设备中 若设备是gpu，就是放到gpu上面训练
    model.to(device)
    optimizer = optim.Adam(model.parameters())

    # 修改optimizer中的learn rate的值，注意力机制的内容
    scheduler = NoamOpt(optimizer, warmup=60, dimension=dim, factor=lr)
    # scheduler = NoamOpt(optimizer, warmup=30, dimension=dim, factor=lr)
    # 获得最好的auc
    best_auc = 0.0
    break_times = 0
    # 进行模型训练的代码 循环epoch次
    for epoch in range(epochs):
        # 如果模型中有Batch Normalization）和Dropout层，就需要使用
        # 因为模型有dropout 所以需要加
        model.train()
        running_loss = 0.0
        scheduler.step()  # 不停调整learning rate
        print("train_loader:{}".format(train_loader.shape))
        # train_bar = tqdm(train_loader)

        # train_bar此时shape(317,3,128,100)
        for data in train_loader:
            # data[0]是input1 data[1]是input3  data[2]是input4
            # logits shape [128,100,64]
            logits = model(data[0].to(device), data[1].to(device))
            # correct shape [128,100,1]
            correct = data[2].float().unsqueeze(-1).to(device)
            # 这是计算二分类交叉熵的损失函数 logits表示样本，correct表示标签值(标签：它最终是属于哪一类)，也就是0或1
            loss = model.loss_function(logits, correct)
            # 完成对模型参数梯度的归零
            optimizer.zero_grad()
            # 对loss进行反向传播，实现梯度下降
            loss.backward()
            # 使用之前loss.backward得到的梯度对参数进行更新 例如learn rate
            optimizer.step()
            running_loss += loss.item()

            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / train_steps))

        # 验证环境的设置,测试集
        model.eval()
        # 不需要梯度调整和优化
        with torch.no_grad():
            predict = model(valid_data[0].to(device), valid_data[1].to(device)).squeeze(-1).to("cpu")
            correct = valid_data[2]
            pred = []
            cort = []
            for i in range(N_val):
                pred.extend(predict[i][unit_list_val[i] - 1:unit_list_val[i]].cpu().numpy().tolist())
                cort.extend(correct[i][unit_list_val[i] - 1:unit_list_val[i]].numpy().tolist())

            # metrics 回归问题常用的均方根误差MSE  mean_squared_error
            rmse = math.sqrt(metrics.mean_squared_error(cort, pred))
            fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1)
            pred = torch.Tensor(pred) > 0.5
            cort = torch.Tensor(cort) == 1
            acc = torch.eq(pred, cort).sum()
            auc = metrics.auc(fpr, tpr)
            # best_auc记录最好的auc值
            if auc > best_auc:
                break_times = 0  # 如果上升一次 表示有转机 则中断次数归0
                best_auc = auc
            else:
                break_times += 1
            print('val_auc: %.3f mse: %.3f acc: %.3f' % (auc, rmse, acc / len(pred)))

    print(best_auc)
    return best_auc
