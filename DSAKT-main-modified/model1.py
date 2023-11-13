import torch.nn as nn
import torch
import numpy as np
import math
from device import return_device
import torch.nn.functional as func

# 可以用torch.cuda.get_device_name(0)查看当前显卡的名称
device = return_device()


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, window_size):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(embed_dim)
        self.scale_attention = ScaledDotProductAttention(dropout, window_size, embed_dim, num_heads)
        self.linear = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=True)

    # qkv.shape batch_size window_size dim
    def forward(self, q, k, v, mask=1):
        bs, tgt_len, _ = q.shape
        # 准备mask层 np.ones按照维度生成1，np.triu，k=1，要保留的上三角部分的偏移量为1，将对角线及下三角矩阵变为0，astype('bool')将非0变为True，0变为False，from_numpy将numpy数组变为pytorch张量
        _mask = np.triu(np.ones((1, 1, tgt_len, tgt_len)), k=mask).astype('uint8')
        attn_mask = (torch.from_numpy(_mask) == 0).to(device)
        # 和attn_mask形状相同的全0张量
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        # masked_fill_原地操作函数，float("-inf")负无穷，new_attn_mask和attn_mask形状相同，attn_mask为True的位置会用负无穷填充到new_attn_mask同样位置，即上三角变为负无穷
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        # 此时attn_mask是一个对角线及下三角为0，上三角为负无穷的张量
        attn_mask = new_attn_mask
        # 缩放点积注意力 attn_output 128 8,100,8
        attn_output = self.scale_attention(q, k, v, attn_mask)
        # 128,100,64
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)
        attn_output = self.linear(attn_output)
        return attn_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout, window_size, embed_dim, num_heads):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.head = num_heads
        self.d_k = int(embed_dim / num_heads)
        # k和v的权重和偏置参数不同
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Drop_out = nn.Dropout(dropout)

    # qkv bs,ws,dim 此时attn_mask为对角及下三角为0，上三角为-inf
    def forward(self, q, k, v, attn_mask):
        bs = q.size(0)
        # view改变张量形状，-1表示未知维度，即总的维度乘积除以已知的维度乘积，transpose交换张量的维度，shape变为bs,head,window_size,d_k每个头的特征维度
        k = self.k_linear(k).view(bs, -1, self.head, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.head, self.d_k).transpose(1, 2)
        q = self.k_linear(q).view(bs, -1, self.head, self.d_k).transpose(1, 2)
        # transpose对最后两个维度进行转置，matmul张量乘法（第一个张量的最后一个维度等于第二个张量的倒数第二个维度，乘积结果是删除刚才的那两个相等的维度，将第二个张量的最后一个维度放在第一个张量的最后一个维度）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 以下这段加上就差 shape(bs,head,ws,ws)
        bs, h, ws, _ = scores.shape

        # torch.no_grad():禁用梯度计算和自动求导，所有的张量操作都不会被记录用于自动求导，从而节省内存并提高运行速度。
        with torch.no_grad():
            # arange创建一个0到ws-1的连续整数序列，expand沿着指定的维度进行复制和扩展 x1.shape(ws,ws)
            x1 = torch.arange(ws).expand(ws, -1).to(device)
            # transpose对张量x1进行转置操作，contiguous并确保转置后的张量在内存中是连续存储的
            x2 = x1.transpose(0, 1).contiguous()
            scores_ = scores + attn_mask
            scores_ = self.softmax(scores_)
            dist_cum_scores = torch.cumsum(scores_, dim=-1)
            dist_total_scores = torch.sum(scores_, dim=-1, keepdim=True)
            # 相对位置的影响
            position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
            dist_scores = (dist_total_scores - dist_cum_scores) * position_effect
            # dist_scores = dist_scores.sqrt().detach()
            '''# -1e32指-1乘以10的32次方 attn_mask对角和下三角为0，attn_mask == 0为True的部分会被第二个参数替代并返回,masked_fill_是masked_fill的原地版
            # scores会将对角线及下三角的数转换为-1e32
            scores_ = scores.masked_fill(attn_mask == 0, -1e32)
            # func.softmax是一个函数，只是对张量进行一次softmax操作，并不需要作为网络的一部分。dim=-1表示最后一个维度，将数据压缩到每行的和为1 shape不变，
            # 对角线及下三角为0，其余每行之和为1，但最后一行每个数字都相等，因为都是-1e32参与计算
            scores_ = func.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            # *点积乘法，shape相同，对应位置元素相乘。结果是上三角全为0，上三角全为-inf,这个有什么意义呢
            scores_ = scores_ * attn_mask.float().to(device)
            # cumsum沿着最后一个维度计算累加和
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, ws, ws
            # sum求和，将最后一个维度求和变成一个数，全部是1
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # bs, 8, ws, 1 全1
            # print(f"distotal_scores: {disttotal_scores}")
            # [None, None, :, :]shape变为[1,1,ws,ws],FloatTensor转为浮点型
            position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
            # bs, 8, ws, ws clamp将最后结果截断，如果有小于min的，就变为min
            dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)
            # 对dist_scores的每个元素都进行开根号
            dist_scores = dist_scores.sqrt().detach()'''
        # total_effect = torch.clamp(dist_scores.exp(), min=1e-5, max=1e5)
        #scores = scores + total_effect
        # scores.masked_fill_(attn_mask == 0, -1e32)
        # scores = scores * dist_scores
        scores = self.softmax(scores)
        scores = self.Drop_out(scores)
        output = torch.matmul(scores, v)
        '''scores.masked_fill_(attn_mask == 0, -1e32)
        scores = func.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        pad_zero = torch.zeros(bs, h, 1, ws).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)'''
        return output
