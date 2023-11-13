import torch


def create_padding_mask(seq):
    """
    创建填充遮罩（Padding Mask）

    参数：
    - seq: 输入序列张量，形状为 (batch_size, seq_length)

    返回：
    - mask: 填充遮罩张量，形状为 (batch_size, 1, 1, seq_length)，其中无效位置为True，有效位置为False
    """
    mask = (seq == 0)  # 将输入序列中值为0的位置视为填充位置
    mask = mask.unsqueeze(1).unsqueeze(2)  # 在维度上扩展，形状变为 (batch_size, 1, 1, seq_length)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻遮罩（Look-ahead Mask）

    参数：
    - size: 输入序列的长度

    返回：
    - mask: 前瞻遮罩张量，形状为 (1, size, size)，其中无效位置为True，有效位置为False
    """
    mask = torch.triu(torch.ones(size, size))  # 生成上三角矩阵
    mask = mask == 0  # 将上三角矩阵中的非零位置视为无效位置
    return mask


# 示例使用
seq = torch.tensor([[1, 2], [3, 4]])  # 假设有一个输入序列
padding_mask = create_padding_mask(seq)
look_ahead_mask = create_look_ahead_mask(seq.size(1))

print("Padding Mask:")
print(padding_mask)

print("Look-ahead Mask:")
print(look_ahead_mask)