from DKT import train_dkt
from SAKT import train_sakt
from DSAKT import train_dsakt
from my_model import train_my_model
import random
import numpy as np
import torch
import csv


def seed_torch(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子


if __name__ == "__main__":
    # 所有模型通用属性'assist09', 'assist12',
    dataset = ['assist15', 'assist12', 'assist09', 'assist17', 'algebra2005_2006']
    # 随机种子 便于重现实验结果
    seed = 42
    seed_torch(seed)
    for ds in dataset:
        train_path = 'data_set/{}/train_data.csv'.format(ds)
        valid_path = 'data_set/{}/test_data.csv'.format(ds)
        print('当前执行的数据集：{}'.format(ds))
        # 获取参数的各种组合
        dkt_best_auc, sakt_best_auc, dsakt_best_auc, my_best_auc = 0, 0, 0, 0
        '''if ds not in ['assist09', 'assist12']:
            # 运行DKT
            window_size = 100
            dim = 200
            batch_size = 128
            max_grad_norm = 20
            epochs = 40
            dkt_best_auc = train_dkt(window_size=window_size,
                                     dim=dim,
                                     train_path=train_path,
                                     valid_path=valid_path,
                                     batch_size=batch_size,
                                     max_grad_norm=max_grad_norm,
                                     epochs=epochs)
            # 运行SAKT
            window_size = 100
            dim = 64
            dropout = 0.2
            heads = 8
            batch_size = 128
            epochs = 40
            sakt_best_auc = train_sakt(window_size=window_size,
                                       dim=dim,
                                       heads=heads,
                                       dropout=dropout,
                                       train_path=train_path,
                                       valid_path=valid_path,
                                       batch_size=batch_size,
                                       epochs=epochs)

            # 运行DSAKT
            lr = 0.3
            dropout = 0.7
            window_size = 100
            dim = 64
            heads = 8
            batch_size = 128
            epochs = 30
            dsakt_best_auc = train_dsakt(window_size=window_size,
                                         dim=dim,
                                         heads=heads,
                                         dropout=dropout,
                                         lr=lr,
                                         train_path=train_path,
                                         valid_path=valid_path,
                                         batch_size=batch_size,
                                         epochs=epochs)
            # 将结果记录到文件
            with open('result/record.csv', mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [2, ds, window_size, dim, batch_size, dkt_best_auc, sakt_best_auc, dsakt_best_auc,
                     my_best_auc])
                     '''
        # 运行my_model
        for window_size in [50, 100, 150, 200]:
            for dim in [16, 32, 64, 128, 256]:
                for batch_size in [32, 64, 128, 256]:
                    lr = 0.3
                    dropout = 0.7
                    heads = 8
                    epochs = 40
                    try:
                        my_best_auc = train_my_model(window_size=window_size,
                                                     dim=dim,
                                                     heads=heads,
                                                     dropout=dropout,
                                                     lr=lr,
                                                     train_path=train_path,
                                                     valid_path=valid_path,
                                                     batch_size=batch_size,
                                                     epochs=epochs)
                    except Exception as e:
                        print(e)
                    # 将结果记录到文件
                    with open('result/record.csv', mode='a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [seed, 2, ds, window_size, dim, batch_size, dkt_best_auc, sakt_best_auc, dsakt_best_auc,
                             my_best_auc])
