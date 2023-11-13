import os
import math
import torch
import argparse
from sklearn import metrics
from utils import getdata
from DSAKT import DSAKT, Encoder, Decoder
from SAKT import SAKT


def predict(window_size: int, model_path: str, data_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_data, N_val, E, unit_list_val = getdata(window_size=window_size, path=data_path, model_type='sakt')

    model = DSAKT(device, E, window_size, 64, 8, 0.7)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        predict = model(pre_data[0].to(device), pre_data[1].to(device)).squeeze(-1).to("cpu")
        correctness = pre_data[2]

        pred = []
        cort = []
        for i in range(N_val):
            pred.extend(predict[i][0:unit_list_val[i]].cpu().numpy().tolist())
            cort.extend(correctness[i][0:unit_list_val[i]].numpy().tolist())

        pred = torch.Tensor(pred) > 0.5
        cort = torch.Tensor(cort) == 1
        acc = torch.eq(pred, cort).sum() / len(pred)

        pred = []
        cort = []
        for i in range(N_val):
            pred.extend(predict[i][unit_list_val[i] - 1:unit_list_val[i]].cpu().numpy().tolist())
            cort.extend(correctness[i][unit_list_val[i] - 1:unit_list_val[i]].numpy().tolist())

        rmse = math.sqrt(metrics.mean_squared_error(cort, pred))
        fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print('val_auc: %.3f mse: %.3f acc: %.3f' % (auc, rmse, acc))


if __name__ == "__main__":
    '''parser = argparse.ArgumentParser()
    parser.add_argument("-ws", "--window_size", required=True)
    parser.add_argument("-d", "--data_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    args = parser.parse_args()'''
    data_path = 'data_set/assist09/assist09_train.csv'
    model_path = 'result/file202208301807_7881.pth'
    window_size = 100
    predict(window_size, model_path, data_path)
