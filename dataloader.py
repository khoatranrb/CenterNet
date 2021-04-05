import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class Data(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.list_file = []
        for label in os.listdir(self.dir):
            list_file = os.listdir(os.path.join(self.dir, label))
            list_file = list(set([i for i in list_file if '.txt' not in i]))
            list_file = [os.path.join(self.dir, label, i) for i in list_file]
            self.list_file += list_file
        self.h, self.w = 256, 256
    def __len__(self):
        return len(self.list_file)
    def __getitem__(self, idx):
        inp_file = self.list_file[idx]
        inp = cv2.resize(cv2.imread(inp_file),(self.w, self.h))
        label_file = '.'.join(inp_file.split('.')[:-1]+['txt'])
        with open(label_file, 'r') as f:
            labels = f.readlines()
        heatmap = np.zeros((self.h//4, self.w//4,2), dtype=np.float32)
        offsetmap = np.zeros((self.h//4, self.w//4,2), dtype=np.float32)
        map3 = np.zeros((self.h//4, self.w//4,2), dtype=np.float32)
        sizemap = np.zeros((self.h//4, self.w//4,1), dtype=np.float32)
        for x in labels:
            label, l, t, w, h = x.split(' ')[:5]
            label, l, t, w, h = int(label), float(l), float(t), float(w), float(h)
            l = l-w/2
            t = t-h/2
            map1 = self.genHeatmap(map1, l, t, w, h, label)
            map2, mask = self.genOffsetmap(map2, l, t, w, h, mask)
            map3 = self.genSizemap(map2, l, t, w, h)
        inp = np.array(inp.transpose((2,0,1))/255.0, dtype=np.float32)
        heatmap = np.array(heatmap.transpose((2,0,1)), dtype=np.float32)
        offsetmap = np.array(offsetmap.transpose((2,0,1)), dtype=np.float32)
        sizemap = np.array(sizemap.transpose((2,0,1)), dtype=np.float32)
        mask = np.array(mask.transpose((2,0,1)), dtype=np.float32)
        return inp, heatmap, offsetmap, sizemap, mask
    def genHeatmap(self, map1, l, t, w, h, label):
        x_c = int(l*self.h/4+w*self.h/8)
        y_c = int(t*self.h/4+h*self.h/8)
        sig_x = max((w*self.h/4)/6, 1)
        sig_y = max((h*self.h/4)/6, 1)
        x_axis = np.array([np.exp(-((i-x_c)**2)/(2*sig_x**2)) for i in range(self.h//4)]).reshape((self.h//4,1))
        y_axis = np.array([np.exp(-((i-y_c)**2)/(2*sig_y**2)) for i in range(self.h//4)]).reshape((self.h//4,1))
        map1[:,:,label] += y_axis.dot(x_axis.T)
        return map1
    def genOffsetmap(self, map1, l, t, w, h, mask):
        x_c = int(l*self.w+w*self.w/2)
        y_c = int(t*self.h+h*self.h/2)
        y_c_float = y_c/4
        x_c_float = x_c/4
        x_c_int = x_c//4
        y_c_int = y_c//4
        map1[y_c_int,x_c_int,0] = y_c_float - y_c_int
        map1[y_c_int,x_c_int,1] = x_c_float - x_c_int
        mask[y_c_int,x_c_int,0] = 1
        return map1, mask
    def genSizemap(self, map1, l, t, w, h):
        x_c_int = int(l*self.w+w*self.w/2)
        y_c_int = int(t*self.h+h*self.h/2)
        x_c_int = x_c_int//4
        y_c_int = y_c_int//4
        map1[y_c_int,x_c_int,1] = w*self.w
        map1[y_c_int,x_c_int,0] = h*self.h
        return map1
