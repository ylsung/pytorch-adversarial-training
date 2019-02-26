import os
import json
import logging

import numpy as np

import torch


def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def one_hot(ids, n_class):
    # --------------------- 
    # author：ke1th 
    # source：CSDN 
    # artical：https://blog.csdn.net/u012436149/article/details/77017832 
    b"""
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """

    assert len(ids.shape) == 1, 'the ids should be 1-D'
    # ids = torch.LongTensor(ids).view(-1,1) 

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)