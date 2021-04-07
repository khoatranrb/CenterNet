from torch.utils.data import DataLoader
from dataloader import Data
import matplotlib.pyplot as plt
import argparse
from models import Net
import torch
from torch.optim import Adam
from tensorboardX import SummaryWriter
import os
from tqdm.auto import tqdm
from losses import CenternetLoss
import sys
from evaluate import eval

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='data/train', help='path to train data')
parser.add_argument('--eval_path', type=str, default='data/val', help='path to evaluate data')
parser.add_argument('--batch', type=int, default=8, help='batch size to train data')
parser.add_argument('--device', type=str, default='cuda', help='use cuda or cpu')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--save_path', type=str, default='saved_model', help='path to saved model')
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--eps', type=float, default=1e-3, help='epsilon to clip output heatmap')
parser.add_argument('--tfboard_path', type=str, default='tfboard/loss', help='path to save tensorboard log')
opt = parser.parse_args()

if os.path.exists(opt.tfboard_path):
    for i in os.listdir(opt.tfboard_path):
        os.remove(os.path.join(opt.tfboard_path, i))

device = opt.device
eps = opt.eps

traindata = Data(opt.train_path)
train_loader = DataLoader(traindata, batch_size=opt.batch, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
valdata = Data(opt.eval_path)
val_loader = DataLoader(valdata, batch_size=1, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)

if len(opt.pretrained)==0:
    model = Net().to(device)
else:
    if opt.device=='cuda':
        model = torch.load(opt.pretrained).to(opt.device)
    else: model = torch.load(opt.pretrained, map_location=torch.device('cpu'))
ctloss = CenternetLoss()

writer = SummaryWriter(logdir=opt.tfboard_path)
optimizer = Adam(params=model.parameters(), lr=opt.lr)

saved_path = opt.save_path
if not os.path.isdir(saved_path): os.mkdir(saved_path)
else:
    for i in os.listdir(saved_path):
        os.remove(os.path.join(saved_path, i))
best = 0
for epoch in tqdm(range(opt.epochs)):
    count_err_file = 0
    model.train()
    torch.cuda.empty_cache()
    train_iter = iter(train_loader)
    c_train = 0
    total_loss_train, total_heat_train, total_off_train, total_size_train = 0, 0, 0, 0
    for i in range(len(train_loader)):
        optimizer.zero_grad()
        try:
            inp, heat, off, size, mask = next(train_iter)
            inp, heat, off, size, mask = inp.to(device), heat.to(device), off.to(device), size.to(device), mask.to(device)
            heat_out, off_out, size_out = model(inp)
            heat_out = torch.clamp(heat_out, eps, 1-eps)
            loss, heat_loss, off_loss, size_loss = ctloss([heat, off, size], [heat_out, off_out, size_out], mask)
            loss.backward()
            optimizer.step()
            c_train += 1
            total_loss_train += loss.item()
            total_heat_train += heat_loss.item()
            total_off_train += off_loss.item()
            total_size_train += size_loss.item()
        except Exception as e:
            # count_err_file+=1
            # sys.stdout.write('\r'+'In epoch {0}, {1} error file(s) has found!'.format(epoch, count_err_file))
            pass
        if i==len(train_loader)-1:
            model.eval()
            total_loss_val, total_heat_val, total_off_val, total_size_val = 0, 0, 0, 0
            val_iter = iter(val_loader)
            c_val = 0
            with torch.no_grad():
                for j in range(len(val_loader)):
                    try:
                        inp, heat, off, size, mask = next(val_iter)
                        inp, heat, off, size, mask = inp.to(device), heat.to(device), off.to(device), size.to(device), mask.to(device)
                        heat_out, off_out, size_out = model(inp)
                        heat_out = torch.clamp(heat_out, eps, 1-eps)
                        loss, heat_loss, off_loss, size_loss = ctloss([heat, off, size], [heat_out, off_out, size_out], mask)
                        total_loss_val += loss.item()
                        total_heat_val += heat_loss.item()
                        total_off_val += off_loss.item()
                        total_size_val += size_loss.item()
                        c_val+=1
                    except Exception as e:
                        # count_err_file+=1
                        # sys.stdout.write('\r'+'In epoch {0}, {1} error file(s) has found!'.format(epoch, count_err_file))
                        pass
            writer.add_scalars('total_loss', {'train':total_loss_train/c_train, 'val':total_loss_val/c_val}, epoch)
            writer.add_scalars('heat_loss', {'train':total_heat_train/c_train, 'val':total_heat_val/c_val}, epoch)
            # writer.add_scalars('off_loss', {'train':total_off_train/c_train, 'val':total_off_val/c_val}, epoch)
            # writer.add_scalars('size_loss', {'train':total_size_train/c_train, 'val':total_size_val/c_val}, epoch)
            iou_obj, thr = eval(opt.eval_path, model_path=model)
            writer.add_scalar('acc_det', iou_obj, epoch)
            # print(iou_obj, thr)
            if iou_obj>best:
                best = iou_obj
                torch.save(model, os.path.join(saved_path, 'model{}.pth'.format(epoch)))
                print('Save model in epoch {0} with accuracy detection {1}.'.format(epoch, best))
