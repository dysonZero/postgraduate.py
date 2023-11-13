import torch.nn as nn
from device import return_device

# 可以用torch.cuda.get_device_name(0)查看当前显卡的名称
device = return_device()


class DKT(nn.Module):
    # window_size:50  num_skills dim外面传
    def __init__(self, window_size, num_skills, dim=64):
        super(DKT, self).__init__()
        self.loss_function = nn.BCELoss()

        self.Embedding = nn.Embedding(num_embeddings=2 * num_skills + 1, embedding_dim=dim, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=dim, hidden_size=dim)
        self.Prediction = nn.Sequential(nn.Linear(in_features=dim, out_features=num_skills + 1),
                                        nn.Sigmoid())

    def forward(self, input, hidden):
        output = self.Embedding(input)
        output = output.permute(1, 0, 2)
        output, (hidden, current) = self.LSTM(output, hidden)
        output = self.Prediction(output.permute(1, 0, 2))
        return output, (hidden.detach(), current.detach())


import math
import torch
import torch.optim as optim
from sklearn import metrics
from utils import getdata, dataloader


def train_dkt(window_size: int, dim: int, train_path: str, valid_path: str, batch_size: int, max_grad_norm: int,
              epochs: int):
    train_data, N_train, E_train, unit_list_train = getdata(window_size=window_size, path=train_path, model_type='sakt')
    valid_data, N_val, E_test, unit_list_val = getdata(window_size=window_size, path=valid_path, model_type='sakt')
    train_loader = dataloader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = dataloader(valid_data, batch_size=batch_size, shuffle=False)
    train_steps = len(train_loader)
    E = max(E_train, E_test)

    model = DKT(window_size=window_size, num_skills=E, dim=dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8)
    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        h = torch.zeros((1, batch_size, dim)).to(device)
        c = torch.zeros((1, batch_size, dim)).to(device)

        # train_bar = tqdm(train_loader)
        for data in train_loader:
            logit, (h, c) = model(data[0].to(device), (h, c))
            logits = torch.gather(logit, 2, data[1].unsqueeze(2).to(device))
            correct = data[2].float().unsqueeze(-1).to(device)

            loss = model.loss_function(logits, correct)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()

            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / train_steps))

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                pred = []
                cort = []

                for batch in range(len(valid_loader)):
                    data = valid_loader[batch]
                    predict, (h, c) = model(data[0].to(device), (h, c))
                    predict = torch.gather(predict.to('cpu'), 2, data[1].unsqueeze(2)).squeeze(-1)
                    correct = data[2]

                    for i in range(batch_size):
                        pos = i + batch_size * batch
                        pred.extend(predict[i][0:unit_list_val[pos]].numpy().tolist())
                        cort.extend(correct[i][0:unit_list_val[pos]].numpy().tolist())

                rmse = math.sqrt(metrics.mean_squared_error(cort, pred))
                fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1)
                pred = torch.Tensor(pred) > 0.5
                cort = torch.Tensor(cort) == 1
                acc = torch.eq(pred, cort).sum()
                auc = metrics.auc(fpr, tpr)
                if auc > best_auc:
                    best_auc = auc
                print('val_auc: %.3f mse: %.3f acc: %.3f' % (auc, rmse, acc / len(pred)))
    print(best_auc)
    return best_auc
