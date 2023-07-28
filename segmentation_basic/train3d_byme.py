from models import Re_mui_net
import torch.optim as optim
from dataloader.MHALoader import mha_data
from torch.utils.data import DataLoader
import torch
from utils.dice_loss import dice_coeff_loss
from utils.train_metrics import metrics3d
import os
import numpy as np

args = {
    'lr': 0.0001,
    'epochs': 4000,
    'snapshot': 100,  # 每100轮保存一下
    'data_path': '/Users/apple/Desktop/科研/脑血管数据集/老数据集',
    'batch_size': 4,
    'ckpt_path': "checkpoint",
    'test_step': 1  # 每几个epoch进行一个model eval
}


def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + 'vnet_Dice' + iter + '.pkl')
    print("{} Saved model to:{}".format("\u2714", args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    net = Re_mui_net().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)  # 权重衰减减少过拟合

    iters = 1
    best_sen, best_dsc = 0., 0.  #
    for epoch in range(args['epochs']):
        net.train()
        train_data = mha_data(args['data_path'], train=True)
        batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=1, shuffle=True)

        for idx, batch in enumerate(batchs_data):
            image = batch[0].type(torch.FloatTensor).cuda()
            label = batch[1].cuda()
            optimizer.zero_grad()

            pred = net(image)

            loss = dice_coeff_loss(pred, label)
            loss.backward()
            optimizer.step()

            ACC, SEN, SPE, IOU, DSC, PRE, AUC = [], [], [], [], [], [], []
            acc, sen, spe, iou, dsc, pre = metrics3d(pred, label, pred.shape[0])
            print(
                '{0:d}:{1:d}] \u25001\u2501\u2501 loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}\tspe:{5:.4f}\tiou:{6:.4f}\tdsc:{7:.4f}\tpre:{8:.4f}'.format
                (epoch + 1, iters, loss.item(), acc / pred.shape[0], sen / pred.shape[0], spe / pred.shape[0],
                 iou / pred.shape[0], dsc / pred.shape[0], pre / pred.shape[0]))
            iters += 1

    # 每个epoch学习率衰减
    adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)
    # 每100轮保存一下
    if (epoch + 1) % args['snapshot'] == 0:
        save_ckpt(net, str(epoch + 1))

    # model eval
    if (epoch + 1) % args['test_step'] == 0:
        test_acc, test_sen, test_spe, test_iou, test_dsc, test_pre, loss2 = model_eval(net, iters)
    if test_sen >= best_sen and (epoch + 1) >= 500:
        save_ckpt(net, "best_SEN")
        best_sen = test_sen
    if test_dsc > best_dsc:
        save_ckpt(net, "best_DSC")
        best_dsc = test_dsc
    print(
        "average ACC:{0:.4f},average SEN:{1:.4f}, average SPE:{2:.4f},average IOU:{3:.4f}, average DSC:{4:.4f},average PRE:{5:.4f}".format(
            test_acc, test_sen, test_spe, test_iou, test_dsc, test_pre))


def model_eval(net, iters):
    print("{}{}{}{}".format(" " * 8, "\u250f", "\u2501" * 61, "\u2513"))
    print("{}{}{}{}".format(" " * 8, "\u2503", " " * 23 + " Start Testing " + " " * 23, "\u2503"))
    print("{}{}{}{}".format(" " * 8, "\u2517", "\u2501" * 61, "\u251b"))
    test_data = mha_data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    ACC, SEN, SPE, IOU, DSC, PRE, AUC = [], [], [], [], [], [], []
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].cuda()
        pred_val = net(image)

        loss = dice_coeff_loss(pred_val, label)
        acc, sen, spe, iou, dsc, pre = metrics3d(pred_val, label, pred_val.shape[0])
        print(
            "---test ACC:{0:.4f} test SEN:{1:.4f} test SPE:{2:.4f} test IOU:{3:.4f} test DSC:{4:.4f} test PRE:{5:.4f}".format
            (acc, sen, spe, iou, dsc, pre))

        ACC.append(acc)
        SEN.append(sen)
        SPE.append(spe)
        IOU.append(iou)
        DSC.append(dsc)
        PRE.append(pre)

    return np.mean(ACC), np.mean(SEN), np.mean(SPE), np.mean(IOU), np.mean(DSC), np.mean(PRE), loss
