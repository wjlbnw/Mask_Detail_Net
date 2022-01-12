import numpy as np
import torch

import os
from torch.utils.data import DataLoader
from torchvision import transforms
from data.data import DIBCODataset, DoubleNetDIBCODataSet
from tool import pyutils
import argparse

from tensorboardX import SummaryWriter

from network.Mask_Detail_Net import Mask_Net
from network.Mask_Detail_Net import Asymmetry_Net
from tool import unet_tool
from dataSetConfig.DibcoDataSet import Dibco2009DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2010DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2011DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2012DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2013DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2014DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2016DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2017DataSetIniter
from dataSetConfig.DibcoDataSet import Dibco2018DataSetIniter
from tool.Best_Weight_Content import Best_Weights_Content


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]

        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - torch.mean(N_dice_eff)

        return loss

class BinaryF_scoreLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryF_scoreLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]


        # 将宽高 reshape 到同一纬度
        input_flat = input.view(-1)
        targets_flat = targets.view(-1)

        TP = (input_flat * targets_flat).sum()
        FP = ((1-targets_flat) * input_flat).sum()
        FN = ((1-input_flat) * targets_flat).sum()
        # 平滑变量
        smooth = 1e-5
        precision = TP / (TP + FP + smooth)
        recall = TP / (TP + FN + smooth)

        f_score = (2 * precision * recall) / (precision + recall + smooth)

        return 1 - f_score


def TP_FP_FN(out, label):

    n, c, h, w = label.size()
    assert n == 1 and c == 1
    n, c, h, w = out.size()
    # print(out.size())
    assert n ==1 and c == 2

    ret = torch.zeros(label.size(), dtype=torch.float32)
    ret[out[:, 0:1, :, :] > out[:, 1:2, :, :]] = 1

    # 维度转换
    y_pred = ret.reshape(-1)
    y_true = label.reshape(-1)

    TP = y_true*y_pred
    FP = (1-y_true)*y_pred
    FN = (1-y_pred)*y_true

    iou = torch.sum(TP) / torch.sum(y_true + y_pred - TP)

    return torch.sum(TP).item(), torch.sum(FP).item(), torch.sum(FN).item(), iou.item()


def list_mean(_list):
    sum = 0
    for a in _list:
        sum += a
    return sum / len(_list)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--save_mode_step", default=2000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=0, type=float)

    parser.add_argument("--session_name", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights",
                        default=None, type=str)

    parser.add_argument("--data_root", default='/home/wujiali/DeepLearning/dataset/DIBCO', type=str)
    parser.add_argument("--tblog_dir", default='train_log', type=str)
    parser.add_argument("--ret_root", default='train_ret', type=str)
    parser.add_argument("--out_root", default='/home/wujiali/DeepLearning/train_out/DIBCO_out', type=str)

    parser.add_argument("--cut_size", default=1024, type=int)

    args = parser.parse_args()



    log_path = os.path.join(args.out_root, args.session_name, )
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    pyutils.Logger(os.path.join(log_path, args.session_name + '.log'))


    print(vars(args))




    tblogger = SummaryWriter(os.path.join(args.out_root, args.session_name, args.ret_root, args.tblog_dir))

    train_dataset = DoubleNetDIBCODataSet(train_set_list=[

        Dibco2010DataSetIniter(os.path.join(args.data_root, 'dibco2010')),
        Dibco2011DataSetIniter(os.path.join(args.data_root, 'dibco2011')),
        Dibco2012DataSetIniter(os.path.join(args.data_root, 'dibco2012')),
        Dibco2013DataSetIniter(os.path.join(args.data_root, 'dibco2013')),
        Dibco2014DataSetIniter(os.path.join(args.data_root, 'dibco2014')),
        Dibco2016DataSetIniter(os.path.join(args.data_root, 'dibco2016')),
        Dibco2017DataSetIniter(os.path.join(args.data_root, 'dibco2017'))
    ], transform=transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                               hue=0.1),
        transforms.Normalize(mean=(0.406, 0.456, 0.485),
                             std=(0.225, 0.224, 0.229))
   ]), transform_concat=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((args.crop_size, args.crop_size), pad_if_needed=True, fill=0.0, padding_mode='constant')
   ]))



    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    val_dataset = DIBCODataset(train_set_list=[
        Dibco2009DataSetIniter(os.path.join(args.data_root, 'dibco2009'))
    ], transform=transforms.Compose([
        transforms.Normalize(mean=(0.406, 0.456, 0.485),
                             std=(0.225, 0.224, 0.229))
   ]), transform_concat=None)

    val_data_loader = DataLoader(val_dataset, shuffle=False, num_workers=1)

    # weights_dict = torch.load(args.weights)
    mask_model = Mask_Net(in_channel=3, out_channel=3)
    # mask_model.load_state_dict(weights_dict['mask_weight'])
    mask_optimizer = torch.optim.Adam(mask_model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wt_dec)
    mask_model = torch.nn.DataParallel(mask_model).cuda()
    mask_model.train()

    seg_model = Asymmetry_Net(in_channel=4)
    # seg_model.load_state_dict(weights_dict['seg_weight'])
    seg_optimizer = torch.optim.Adam(seg_model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.wt_dec)
    seg_model = torch.nn.DataParallel(seg_model).cuda()
    seg_model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_mask', 'loss_seg')

    global_step = 0
    timer = pyutils.Timer("Session started: ")
    dice_loss = BinaryDiceLoss()
    f_socre_loss = BinaryF_scoreLoss()
    best_f_score_mean = 0
    best_f_score_weights = Best_Weights_Content(10)
    best_iou_mean = 0

    for ep in range(args.max_epoches):

        for iter, (img, gt, mask) in enumerate(train_data_loader):

            mask_out = mask_model(img)
            # mask_out = visualization.max_norm(mask_out)
            bg = torch.ones_like(gt, dtype=torch.float32)
            bg[mask == 1.] = 0
            bg[gt == 1.] = 0
            mask_label = torch.concat((mask, gt, bg), dim=1).cuda(non_blocking=True)

            if global_step < 10000:
                loss_mask = dice_loss(mask_out, mask_label)
            else:
                loss_mask = torch.tensor(0, dtype=torch.float32)
            concat_in = torch.concat((img.cuda(non_blocking=True), mask_out[:, 0:1, :, :]), dim=1)
            seg_out = seg_model(concat_in)
            seg_label = torch.concat((gt, 1-gt), dim=1).cuda(non_blocking=True)

            loss_seg = f_socre_loss(seg_out, seg_label)

            loss = loss_mask + loss_seg

            mask_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            loss.backward()
            seg_optimizer.step()
            mask_optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_mask': loss_mask.item(), 'loss_seg': loss_seg.item()})
            global_step += 1

            if global_step % 50 == 0:
                timer.update_progress(global_step / max_step)

                print('Iter:%5d/%5d' % (global_step , max_step),
                      'loss:%.4f %.4f %.4f' % avg_meter.get('loss', 'loss_mask', 'loss_seg'),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),

                      # 'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      flush=True)

                avg_meter.pop()

                itr = global_step
                loss_dict = {'loss': loss.item(), 'loss_mask': loss_mask.item(), 'loss_seg': loss_seg.item()}

                tblogger.add_scalars('loss', loss_dict, itr)

                mask_model.eval()
                seg_model.eval()
                cut_size = args.cut_size
                with torch.no_grad():
                    f_score_list = []
                    precision_list = []
                    recall_list = []
                    iou_list = []

                    for iter, (img, gt) in enumerate(val_data_loader):
                        n, c, h, w = img.size()

                        seg_ret = torch.zeros(size=(n,2,h,w), dtype=torch.float32)
                        for top in range(0, h, cut_size):
                            for left in range(0, w, cut_size):
                                pic_in_window = img[:, :, top:min(top + cut_size, h), left:min(left + cut_size, w)]

                                N, C, window_h, window_w = pic_in_window.size()
                                padding_pic, pad_top, pad_left = unet_tool.pad_img(pic_in_window)
                                mask_out = mask_model(padding_pic)
                                mask_out = mask_out[:, 0:1, :, :]
                                seg_out = seg_model(torch.concat((padding_pic.cuda(), mask_out), dim=1))
                                seg_out = seg_out[:, :, pad_top:pad_top+window_h, pad_left:pad_left+window_w].cpu()
                                seg_ret[:, :, top: top+window_h, left: left+window_w] = seg_out[:, :, :, :]

                        TP_sum, FP_sum, FN_sum, iou = TP_FP_FN(seg_ret, gt)
                        precision = (TP_sum + 1e-5) / (TP_sum + FP_sum + 1e-5)
                        recall = (TP_sum + 1e-5) / (TP_sum + FN_sum + 1e-5)
                        f_score = (2*precision*recall + 1e-5)/(precision + recall + 1e-5)

                        precision_list.append(precision)
                        recall_list.append(recall)
                        f_score_list.append(f_score)
                        iou_list.append(iou)

                    f_score_mean = list_mean(f_score_list)
                    precision_mean = list_mean(precision_list)
                    recall_mean = list_mean(recall_list)
                    iou_mean = list_mean(iou_list)
                    print('val result F-socre: %.4f, precision: %.4f, recall: %.4f' % (f_score_mean, precision_mean, recall_mean),
                          'Fin:%s' % (timer.str_est_finish()))
                    best_f_score_weights.insert(score=f_score_mean,
                                                weigth_dict={
                                                    'mask_weight': mask_model.module.state_dict(),
                                                    'seg_weight': seg_model.module.state_dict()
                                                },
                                                weight_path=os.path.join(args.out_root, args.session_name, args.ret_root,
                                                                   args.session_name + '_best_fscore_iter_%05d.pth'%(global_step)))


                    if f_score_mean > best_f_score_mean:
                        best_f_score_mean = f_score_mean
                        dict = {
                            'mask_weight': mask_model.module.state_dict(),
                            'seg_weight': seg_model.module.state_dict()
                        }
                        torch.save(dict, os.path.join(args.out_root, args.session_name, args.ret_root,
                                                      args.session_name + '_best_f_score.pth'))
                    if iou_mean > best_iou_mean:
                        best_iou_mean = iou_mean
                        dict = {
                            'mask_weight': mask_model.module.state_dict(),
                            'seg_weight': seg_model.module.state_dict()
                        }
                        torch.save(dict, os.path.join(args.out_root, args.session_name, args.ret_root,
                                                      args.session_name + '_best_iou.pth'))

                    val_dict = {
                        'val_f_score': f_score_mean,
                        'val_precision': precision_mean,
                        'val_recall': recall_mean,
                        'IOU' : iou_mean
                    }
                    tblogger.add_scalars('val', val_dict, itr)

                mask_model.train()
                seg_model.train()


            if global_step % args.save_mode_step == 0:
                print('')
                timer.reset_stage()
                dict={
                    'mask_weight':mask_model.module.state_dict(),
                    'seg_weight' :seg_model.module.state_dict()
                }
                torch.save(dict, os.path.join(args.out_root, args.session_name, args.ret_root,
                                                                   args.session_name + '_iter_newst.pth'))
        else:
            timer.reset_stage()

    dict = {
        'mask_weight': mask_model.module.state_dict(),
        'seg_weight': seg_model.module.state_dict()
    }
    torch.save(dict, os.path.join(args.out_root, args.session_name, args.ret_root,  args.session_name + '.pth'))
    print('Session end...')

