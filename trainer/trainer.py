import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


def Trainer(data_type, epochs, model, model_optimizer, train_dl,  test_dl, device, logger, experiment_log_dir):
    # Start training and testing
    logger.debug("Training and Testing started ....")

    criterion = nn.NLLLoss()

    max_test_auc = 0
    max_test_acc = 0
    max_test_precision = 0
    max_test_recall = 0
    max_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train and validate
        train_loss, train_acc, train_auc, train_precision, train_recall = model_train(model, model_optimizer, criterion, train_dl, device)
        test_loss, test_acc, test_auc, test_precision, test_recall, _, _ = model_evaluate(model, test_dl, device)

        if max_test_auc <= test_auc:
            max_epoch = epoch
            max_test_acc = test_acc
            max_test_auc = test_auc
            max_test_precision = test_precision
            max_test_recall = test_recall

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss : {train_loss:.4f}\t | \t Accuracy : {train_acc:2.4f} | \t AUC : {train_auc:2.4f} | \t Precision : {train_precision:2.4f} | \t Recall : {train_recall:2.4f}\n'
                     f'Test Loss : {test_loss:.4f}\t | \t Accuracy : {test_acc:2.4f} | \t AUC : {test_auc:2.4f} | \t Precision : {test_precision:2.4f} | \t Recall : {test_recall:2.4f}')

        if epoch == 20:
            os.makedirs(os.path.join(experiment_log_dir, "saved_models" + str(epoch)), exist_ok=True)
            chkpoint = {'model_state_dict': model.state_dict()}

            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models" + str(epoch), f'ckp_last.pt'))
    with open("experiments_logs(CRBPSA)/result_all.txt",'a',encoding='utf-8') as f :
        f.write("dataset:{}\n".format(data_type))
        f.write(f'epoch:{max_epoch}\t | \tmax_test_acc: {max_test_acc:2.4f} \t | \tmax_test_auc: {max_test_auc:2.4f} | \tmax_test_precision: {max_test_precision:2.4f} | \tmax_test_recall: {max_test_recall:2.4f}\n')

    logger.debug("\n################## Training and Testing is Done! #########################")


def model_train(model, model_optimizer, criterion, train_dl, device):
    total_loss = []
    total_acc = []
    total_auc = []
    total_precision = []
    total_recall = []
    model.train()

    for batch_idx, (data, labels) in enumerate(train_dl):

        data = data.to(device)
        labels = labels.long().to(device)

        # optimizer
        model_optimizer.zero_grad()

        yt = model(data)

        # compute loss
        loss = criterion(yt, labels)

        auc = roc_auc_score(labels.cpu(), yt.detach().cpu()[:,1])
        acc = accuracy_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
        precision = precision_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
        recall = recall_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))

        total_acc.append(acc)
        total_precision.append(precision)
        total_recall.append(recall)
        total_auc.append (auc)

        total_loss.append(loss.item())
        loss.backward()

        #更新参数
        model_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_precision = torch.tensor(total_precision).mean()
    total_recall = torch.tensor(total_recall).mean()

    return total_loss, total_acc, total_auc, total_precision, total_recall


def model_evaluate(model, test_dl, device):
    model.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_precision = []
    total_recall = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels in test_dl:
            data = data.to(device)
            labels = labels.long().to(device)

            yt = model(data)

            # compute loss
            loss = criterion(yt, labels)
            auc = roc_auc_score(labels.cpu(), yt.detach().cpu()[:,1])
            acc = accuracy_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
            precision = precision_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
            recall = recall_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))

            total_loss.append(loss.item())
            total_acc.append(acc)
            total_precision.append(precision)
            total_recall.append(recall)
            total_auc.append (auc)

            pred = yt.max(1, keepdim=True)[1]
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_auc = torch.tensor(total_auc).mean()
    total_precision = torch.tensor(total_precision).mean()
    total_recall = torch.tensor(total_recall).mean()

    return total_loss, total_acc, total_auc, total_precision, total_recall, outs, trgs
