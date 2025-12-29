import torch
from torch import Tensor
from typing import Tuple


# class Metrics:
#     def __init__(self, num_classes: int, ignore_label: int, device) -> None:
#         self.ignore_label = ignore_label
#         self.num_classes = num_classes
#         self.hist = torch.zeros(num_classes, num_classes).to(device)
#
#     def update(self, pred: Tensor, target: Tensor) -> None:
#         pred = pred.argmax(dim=1)
#         keep = target != self.ignore_label
#         self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)
#
#     def compute_iou(self) -> Tuple[Tensor, Tensor]:
#         ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
#         ious[ious.isnan()]=0.
#         miou = ious.mean().item()
#         # miou = ious[~ious.isnan()].mean().item()
#         ious *= 100
#         miou *= 100
#         return ious.cpu().numpy().round(2).tolist(), round(miou, 2)
#
#     def compute_f1(self) -> Tuple[Tensor, Tensor]:
#         f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
#         f1[f1.isnan()]=0.
#         mf1 = f1.mean().item()
#         # mf1 = f1[~f1.isnan()].mean().item()
#         f1 *= 100
#         mf1 *= 100
#         return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)
#
#     def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
#         acc = self.hist.diag() / self.hist.sum(1)
#         acc[acc.isnan()]=0.
#         macc = acc.mean().item()
#         # macc = acc[~acc.isnan()].mean().item()
#         acc *= 100
#         macc *= 100
#         return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

import numpy as np
class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device):


        self.n_classes = num_classes
        # self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

        self.ignore_idx = ignore_label

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.argmax(dim=1)
        # pred = pred.squeeze()
        # gt = gt.squeeze()
        valid = (gt != self.ignore_idx)

        for i_part in range(0, self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    # def compute_iou(self) -> Tuple[Tensor, Tensor]:
    #     ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
    #     ious[ious.isnan()]=0.
    #     miou = ious.mean().item()
    #     # miou = ious[~ious.isnan()].mean().item()
    #     ious *= 100
    #     miou *= 100
    #     return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_iou(self, verbose=True):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        # eval_result = dict()
        # eval_result['jaccards_all_categs'] = jac
        # eval_result['mIoU'] = np.mean(jac) * 100
        ious = jac
        for iou in ious:
            iou = round(iou*100, 2)
        return ious, round(np.mean(jac) * 100, 2)
        # if verbose:
        #     print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(eval_result['mIoU']))
        #     class_IoU = jac  # eval_result['jaccards_all_categs']
        #     for i in range(len(class_IoU)):
        #         spaces = ''
        #         for j in range(0, 20 - len(self.cat_names[i])):
        #             spaces += ' '
        #         print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))
        #
        # return eval_result

