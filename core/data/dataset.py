from __future__ import print_function

import torch
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import argparse
import os
from PIL import Image


class MetaMSTAR(torch.utils.data.Dataset):
    """MSTAR Dataset for meta-based few-shot SAR image classification.

    Args:
        root (string): Root directory of dataset where ``MSTAR`` exists.
        train (bool, optional): If True, creates dataset from ``meta_train``,
            otherwise from ``meta_test``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    training_folder = 'meta_train'
    training_categories = ['BMP2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131']
    test_folder = 'meta_test'
    test_categories = ['2S1', 'BRDM2', 'ZSU234']

    def __init__(
            self,
            args,
            root,
            train=True,
            transform=None,
            method_type=None
    ):
        super(MetaMSTAR, self).__init__()
        self.method_type = method_type
        self.root = root
        self.train = train
        if train:
            self.num_tasks = args['train_episode']
            self.num_ways = args['train_way']
            self.num_shots = args['train_shot']
            self.num_queries = args['train_query']
        else:
            self.num_tasks = args['test_episode']
            self.num_ways = args['test_way']
            self.num_shots = args['test_shot']
            self.num_queries = args['test_query']
        self.depression_angle = args['depression_angle']

        # transform need to be defined outside
        assert transform is not None
        self.transform = transform

        if self.train:
            if method_type == 'preTrain':
                pass
            else:
                self.num_categories = len(self.training_categories)
                self.data_dir = os.path.join(self.root, self.training_folder)
                self.lab2img_paths = dict()
                for i, category in enumerate(self.training_categories):
                    img_dir = os.path.join(self.data_dir, category, str(self.depression_angle))
                    for _, _, files in os.walk(img_dir):
                        paths = [os.path.join(img_dir, file) for file in files]
                        self.lab2img_paths[i] = paths
        else:
            self.num_categories = len(self.test_categories)
            self.data_dir = os.path.join(self.root, self.test_folder)
            self.lab2img_paths = dict()
            for i, category in enumerate(self.test_categories):
                img_dir = os.path.join(self.data_dir, category, str(self.depression_angle))
                for _, _, files in os.walk(img_dir):
                    paths = [os.path.join(img_dir, file) for file in files]
                    self.lab2img_paths[i] = paths

    def __getitem__(self, index):
        if self.method_type == 'preTrain':
            pass
        else:
            # np.random.seed(index)
            support_xs = []
            support_ys = []
            query_xs = []
            query_ys = []
            # sample class index for the task
            cat_sampled = np.random.choice(self.num_categories, self.num_ways, replace=False)
            for i, cat in enumerate(cat_sampled):
                # sample image index for the support set and the query set
                img_paths = self.lab2img_paths[cat]
                indices_sampled = np.random.choice(
                    len(img_paths), self.num_shots + self.num_queries, replace=False)
                support_ids, query_ids = np.split(indices_sampled, [self.num_shots])
                # construct the support set
                for j in support_ids:
                    img = Image.open(img_paths[j])
                    img = self.transform(img)
                    support_xs.append(img)
                    support_ys.append(i)
                # construct the query set
                for j in query_ids:
                    img = Image.open(img_paths[j])
                    img = self.transform(img)
                    query_xs.append(img)
                    query_ys.append(i)
            support_xs, support_ys = torch.stack(support_xs).repeat(1, 3, 1, 1), torch.tensor(support_ys)
            query_xs, query_ys = torch.stack(query_xs).repeat(1, 3, 1, 1), torch.tensor(query_ys)
            image_size = support_xs.size(2)
            support_xs = support_xs.view(self.num_ways, self.num_shots, 3, image_size, image_size)
            query_xs = query_xs.view(self.num_ways, self.num_queries, 3, image_size, image_size)
            x = torch.cat([support_xs, query_xs], dim=1)
            return x

    def __len__(self):
        if self.method_type == 'preTrain':
            pass
        else:
            return self.num_tasks
