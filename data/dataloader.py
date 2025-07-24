import torch
import os
import numpy as np
import torchvision.transforms as transforms

import data.mytransforms as mytransforms
from .dataset import LaneClsDataset

def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, num_lanes, row_anchor):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])

    if dataset == 'Carla':
        train_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'train_gt.txt'),
                                       img_transform=img_transform,
                                       target_transform=target_transform,
                                       simu_transform=None,
                                       griding_num=griding_num, 
                                       row_anchor=row_anchor,
                                       segment_transform=segment_transform,
                                       use_aux=use_aux, 
                                       num_lanes=num_lanes)
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    print(f"Number of training data: {len(train_dataset)}")
    sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)

    return train_loader, cls_num_per_lane

def get_test_loader(batch_size, data_root, dataset):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/test.txt'),img_transform = img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'test.txt'), img_transform = img_transforms)
        cls_num_per_lane = 56

    sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)
    return loader