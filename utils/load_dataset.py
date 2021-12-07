import torch
from torch.utils.data import Dataset, Subset

def load_dataset(cfg):
    pass



def split_dataset(cfg, dataset : Dataset) -> (Subset, Subset, Subset):
    '''
    TODO
    :param cfg:
    :param dataset:
    :return:
    '''
    train_split, val_split = cfg.get("split", [0.7, 0.1])
    dataset_length = len(dataset)
    train_length, val_length = int(train_split * dataset_length), int(val_split * dataset_length)
    test_length = dataset_length - train_length - val_length
    return torch.utils.data.random_split(dataset, [train_length, val_length, test_length])