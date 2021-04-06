from torch.utils.data import DataLoader
from dataloader import Data
import matplotlib.pyplot as plt
import args
from models import Net
import torch
from torch.optim import Adam
from tensorboardX import SummaryWriter
import os
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='task1', help='path to train data')
parser.add_argument('--eval_path', type=str, default='task1', help='path to evaluate data')
parser.add_argument('--batch', type=int, default=8, help='batch size to train data')
parser.add_argument('--device', type=str, default='cuda', help='use cuda or cpu')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--save_path', type=str, default='saved_model', help='path to saved model')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--eps', type=float, default=1e-3, help='epsilon to clip output heatmap')
parser.add_argument('--tfboard_path', type=str, default='tfboard/loss', help='path to save tensorboard log')
opt = parser.parse_args()

if os.path.exists(opt.tfboard_path):
    os.remove(opt.tfboard_path)

traindata = Data(opt.train_path)
train_loader = DataLoader(traindata, batch_size=opt.batch, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
valdata = Data(opt.eval_path)
val_loader = DataLoader(valdata, batch_size=1, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)

model = Net().to(opt.device)
ctloss = CenternetLoss()

writer = SummaryWriter(logdir=opt.tfboard_path)
optimizer = Adam(params=model.parameters(), lr=opt.lr)

saved_path = opt.save_path
if not os.path.isdir(saved_path): os.mkdir(saved_path)

for epoch in tqdm(range(opt.epochs)):
    model.train()
    torch.cuda.empty_cache()
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        optimizer.zero_grad()
        try:
            inp, heat, off, size, mask = next(train_iter)
            inp, heat, off, size, mask = inp.to(device), heat.to(device), off.to(device), size.to(device), mask.to(device)
            heat_out, off_out, size_out = model(inp)
            heat_out = torch.clamp(heat_out, eps, 1-eps)
            loss, heat_loss, off_loss, size_loss = ctloss([heat, off, size], [heat_out, off_out, size_out], mask)
            loss.backward()
            opt.step()
        except Exception as e:
            pass
        if i==len(train_loader)-1:
            model.eval()
            total_loss, total_heat, total_off, total_size = 0, 0, 0, 0
            val_iter = iter(val_loader)
            c = 0
            with torch.no_grad():
                for j in range(len(val_loader)):
                    try:
                        inp, heat, off, size, mask = next(val_iter)
                        inp, heat, off, size, mask = inp.to(device), heat.to(device), off.to(device), size.to(device), mask.to(device)
                        heat_out, off_out, size_out = model(inp)
                        heat_out = torch.clamp(heat_out, eps, 1-eps)
                        loss, heat_loss, off_loss, size_loss = ctloss([heat, off, size], [heat_out, off_out, size_out], mask)
                        total_loss += loss.item()
                        total_heat += heat_loss.item()
                        total_off += off_loss.item()
                        total_size += size_loss.item()
                        c+=1
                    except Exception as e:
                        pass
            if total_loss/c<best_loss:
                best_loss = total_loss/c
                torch.save(model, os.path.join(saved_path, 'model{}.pth'.format(epoch)))
                print('Save model in epoch {0} with loss {1}.'.format(epoch, best_loss))
            writer.add_scalar('total_loss', total_loss/c, epoch)
            writer.add_scalar('heat_loss', total_heat/c, epoch)
            writer.add_scalar('off_loss', total_off/c, epoch)
            writer.add_scalar('size_loss', total_size/c, epoch)
