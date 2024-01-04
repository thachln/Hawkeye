import logging
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt

class PerformanceMeter(object):
    """Record the performance metric during training
    """

    def __init__(self, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        self.current_value = self.values[-1]
        self.best_value = self.best_function(self.values)
        self.best_epoch = self.values.index(self.best_value)

    @property
    def value(self):
        return self.values[-1]


class AverageMeter(object):
    """Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        super(TqdmHandler, self).__init__()

    def emit(self, msg):
        msg = self.format(msg)
        tqdm.write(msg)
        time.sleep(1)


class Timer(object):

    def __init__(self):
        self.start = time.time()
        self.last = time.time()

    def tick(self, from_start=False):
        this_time = time.time()
        if from_start:
            duration = this_time - self.start
        else:
            duration = this_time - self.last
        self.last = this_time
        return duration


def build_config_from_dict(_dict):
    cfg = CN()
    for key in _dict:
        cfg[key] = _dict[key]
    return cfg


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def map_generate(attention_map, pred, p1, p2):
    batches, feaC, feaH, feaW = attention_map.size()

    out_map=torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        map_tpm = attention_map[batch_index]
        map_tpm = map_tpm.reshape(feaC, feaH*feaW)
        map_tpm = map_tpm.permute([1, 0])
        p1_tmp = p1.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p1_tmp)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = p2[pred_ind].unsqueeze(1)

        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p2_tmp)
        out_map[batch_index] = map_tpm.reshape(feaH, feaW)

    return out_map

def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images



def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()

    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta

        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images



def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)
