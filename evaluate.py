import numpy as np
import os
def iou(bb_gt, bb_pred):
    cx_gt, cy_gt, w_gt, h_gt = bb_gt
    cx_pred, cy_pred, w_pred, h_pred = bb_pred

    l_gt = cx_gt - w_gt/2
    l_pred = cx_pred - w_pred/2
    t_gt = cy_gt - h_gt/2
    t_pred = cy_pred - h_pred/2

    l = max(l_gt, l_pred)
    t = max(t_gt, t_pred)
    r = min(l_gt+w_gt, l_pred+w_pred)
    b = min(t_gt+h_gt, t_pred+h_pred)
    if l>=r or t>=b: return 0
    return (r-l)*(b-t) / (h_gt*w_gt + h_pred*w_pred - (r-l)*(b-t))

def eval(dir=None, infer=Inference('saved_model/model199.pth', 'cpu', 'saved'), num_class=2, iou_thr=[0.5, 0.6]):
    res_list = [[0,0,0] for i in range(len(iou_thr))]
    all_file = []
    for label in os.listdir(dir):
        list_file = os.listdir(os.path.join(dir, label))
        list_file = list(set([i for i in list_file if '.txt' not in i]))
        list_file = [os.path.join(dir, label, i) for i in list_file]
        all_file += list_file
    for inp_path in tqdm(all_file):
        # inp_path = '/content/drive/MyDrive/Eastg8/caries/caries/x ray labled/task 1/Exposed pulp X-Rays/001765.png'
        try:
            bb_pred = infer(inp_path, save=False)
            label_file = '.'.join(inp_path.split('.')[:-1]+['txt'])
            with open(label_file, 'r') as f:
                labels = f.readlines()
        except: continue
        bb_gt = [[] for i in range(num_class)]
        for x in labels:
            label, l, t, w, h = x.split(' ')[:5]
            label, l, t, w, h = int(label), float(l), float(t), float(w), float(h)
            l = l-w/2
            t = t-h/2
            bb_gt[label].append([l,t,w,h])
        num_gt = 0
        for i in range(num_class):
            num_gt += len(bb_gt[i])
        for i in range(len(iou_thr)):
            res_list[i][0] += num_gt
        # print(bb_gt)
        # print(bb_pred)
        # print(0)
        for i in range(num_class):
            # # for gt, pred in zip(bb_gt[i], bb_pred[i]):
            # print(gt)
            if len(bb_gt[i])==0:
                for i_iou in range(len(iou_thr)):
                    res_list[i_iou][2]+=len(bb_pred[i])
                continue
            if len(bb_pred[i])==0: continue
            check_gt = [[[0 for ii in range(len(bb_gt[i]))] for jj in range(len(bb_pred[i]))] for kk in range(len(iou_thr))]
            # print(len(bb_gt[i]),len(bb_pred[i]))
            for i_pred, b_pred in enumerate(bb_pred[i]):
                for i_gt, b_gt in enumerate(bb_gt[i]):
                    # print(b_gt, b_pred)
                    iou_res = iou(b_gt, b_pred)
                    for i_iou, iou_ in enumerate(iou_thr):
                        if iou_res>iou_:
                            check_gt[i_iou][i_pred][i_gt] = 1
            check_gt = np.array(check_gt)
            for i_iou in range(len(iou_thr)):
                for i_gt in range(len(bb_gt[i])):
                    # print(check_gt.shape, bb_gt, bb_pred)
                    x = check_gt[i_iou, :, i_gt]
                    if np.sum(x)>0: res_list[i_iou][1]+=1
                for i_pred in range(len(bb_pred[i])):
                    x = check_gt[i_iou, i_pred, :]
                    if np.sum(x)==0: res_list[i_iou][2]+=1
    best_iou_obj, best_thr = -1, -1
    for i_iou, res_iou in enumerate(res_list):
        iou_obj = res_iou[1]/(res_iou[0]+res_iou[2])
        if iou_obj > best_iou_obj:
            best_iou_obj = iou_obj
            best_thr = iou_thr[i_iou]
    print(best_iou_obj, best_thr)
