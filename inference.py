import os
import cv2
import torch
import numpy as np

class Inference():
    def __init__(self, model, device, save_path=''):
        self.device = device
        if model:
            try:
                self.net = model.to(device)
            except:
                try:
                    self.net = torch.load(model).cuda()
                except: self.net = torch.load(model, map_location=torch.device('cpu'))
        self.save_path = save_path
        if not os.path.isdir(save_path): os.mkdir(save_path)
    def preprocess(self, img):
        img = cv2.resize(img, (256, 256))
        img = img.transpose((2,0,1))/255.0
        img = img[np.newaxis, ...].astype(np.float32)
        return torch.Tensor(img).to(self.device)
    def getbb(self, heatmap, offmap, sizemap, thr=0.5):
        def ismax(heat, y, x):
            h, w = heat.shape[:2]
            if y in [h-1,0] or x in [w-1,0]:
                return 0
            if np.argmax(heat[y-1:y+2, x-1:x+2])==4:
                return 1
            return 0
        def get_center(heat, thr=0.5):
            cand = np.where(heat>0.5)
            out = []
            for y, x in zip(list(cand[0]), list(cand[1])):
                if ismax(heat, y, x): out.append([y,x])
            return out
        out = []
        for i in range(heatmap.shape[-1]):
            heat = heatmap[:,:,i]
            cands = get_center(heat)
            res = []
            for j in range(len(cands)):
                cand = cands[j]
                pos = cand.copy()
                print(offmap.shape, pos)
                cand[1] = int((offmap[pos[0], pos[1], 0]+pos[0])*4)/256
                cand[0] = int((offmap[pos[0], pos[1], 1]+pos[1])*4)/256
                cand.append(int(sizemap[pos[0], pos[1], 1])/256)
                cand.append(int(sizemap[pos[0], pos[1], 0])/256)
                cand[1]-=cand[3]/2
                cand[0]-=cand[2]/2
                res.append(cand)
            out.append(res)
        return out

    def __call__(self, path=None, test=None):
        if not test:
            img = cv2.imread(path)
            inp = self.preprocess(img)
            with torch.no_grad():
                heat, off, size = self.net(inp)
            if self.device == 'cuda':
                heat = (heat.cpu().detach().numpy()[0]).transpose((1,2,0))
                off = (off.cpu().detach().numpy()[0]).transpose((1,2,0))
                size = (size.cpu().detach().numpy()[0]).transpose((1,2,0))
            else:
                heat = (heat.numpy()[0]).transpose((1,2,0))
                off = (off.numpy()[0]).transpose((1,2,0))
                size = (size.numpy()[0]).transpose((1,2,0))
            outname = path.split('/')[-1]
        else:
            img, heat, off, size = test
            outname = 'out.png'
        bb = self.getbb(heat, off, size)
        if len(self.save_path)==0: return bb
        color_init = [0,0,0]
        H, W = img.shape[:2]
        im = img.copy()
        for i in range(2):
            bbx = bb[i]
            color = color_init.copy()
            color[i] = 255
            for cy, cx, h, w in bbx:
                h = int(h*H)
                w = int(w*W)
                l = int(cx*W)-w//2
                t = int(cy*H)-h//2
                im = cv2.rectangle(im, (l,t), (l+w,t+h), color, 3)
        cv2.imwrite(os.path.join(self.save_path, outname), im)
