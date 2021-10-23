# -*- coding: utf-8 -*-
import datetime
import os
from time import time

import torch
from core.data import get_dataloader
from core.utils import model_dict, init_seed, get_best_file
from core import backbone
from core.methods.baselinetrain import BaselineTrain
from core.methods.protonet import ProtoNet
from core.methods.matchingnet import MatchingNet
from core.methods.relationnet import RelationNet


class Trainer(object):
    """
    The trainer.

    Build a trainer from config dict, set up optimizer, model, etc. Train/test/val and log.
    """

    def __init__(self, config):
        self.config = config
        init_seed(config["seed"], config["deterministic"])
        self.device = torch.device("cuda:0")  # for one gpu, multi-gpus in the newer version

        self.model = self._init_model(config)
        self.train_loader, self.test_loader = self._init_dataloader(config)
        if config['optimization'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            raise ValueError('Unknown optimization, please define by yourself')

        experi = config['model'] + '_trWay_' + str(config['train_way']) + '_trShot_' + str(config['train_shot'])
        self.checkpoint_dir = os.path.join(os.path.join(config['save_path'], config['method']), experi)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.from_epoch = config['start_epoch']
        if config['resume']:
            load_dir = config['resume_path']  # 'xxx_epoch_n.tar'
            resume_file = get_best_file(load_dir)
            tmp = torch.load(resume_file)
            self.model.load_state_dict(tmp['state'])
            self.from_epoch = tmp['epoch'] + 1

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        if config['method'] == 'baseline':
            model = BaselineTrain(model_dict[config['model']], config['num_class'])
        elif config['method'] == 'prototype':
            train_few_shot_params = dict(n_way=config['train_way'], n_support=config['train_shot'])
            model = ProtoNet(model_dict[config['model']], **train_few_shot_params)
        elif config['method'] == 'matching':
            train_few_shot_params = dict(n_way=config['train_way'], n_support=config['train_shot'])
            model = MatchingNet(model_dict[config['model']], **train_few_shot_params)
        elif config['method'] == 'relation':
            train_few_shot_params = dict(n_way=config['train_way'], n_support=config['train_shot'])
            if config['model'] == 'Conv4':
                feature_model = backbone.Conv4NP
            else:
                feature_model = lambda: model_dict[config['model']](flatten=False)
            loss_type = 'mse' if config['is_mse'] else 'softmax'
            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        else:
            raise NotImplementedError
        model = model.to(self.device)
        return model

    def _init_dataloader(self, config):
        """
        Init dataloaders.(train_loader, val_loader and test_loader)

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (train_loader, val_loader and test_loader).
        """
        train_loader = get_dataloader(config, "train")
        test_loader = get_dataloader(config, "test")

        return train_loader, test_loader

    def _cal_time_scheduler(self, start_time, epoch_idx):
        """
        Calculate the remaining time and consuming time of the training process.

        Returns:
            str: A string similar to "00:00:00/0 days, 00:00:00". First: comsuming time; Second: total time.
        """
        total_epoch = self.config["stop_epoch"] - self.from_epoch - 1
        now_epoch = epoch_idx - self.from_epoch

        time_consum = datetime.datetime.now() - datetime.datetime.fromtimestamp(start_time)
        time_consum -= datetime.timedelta(microseconds=time_consum.microseconds)
        time_remain = (time_consum * (total_epoch - now_epoch)) / (now_epoch)

        res_str = str(time_consum) + "/" + str(time_remain + time_consum)

        return res_str

    def train_loop(self):
        """
        The normal train loop: train-val-test and save model when val-acc increases.
        """
        best_test_acc = float("-inf")
        experiment_begin = time()
        for epoch_idx in range(self.from_epoch + 1, self.config["stop_epoch"]):
            print("============ Train on the train set ============")
            self._train(epoch_idx)
            print("============ Testing on the test set ============")
            acc_mean = self._test()
            print(" * Acc@1 {:.3f} Best acc {:.3f}".format(acc_mean, best_test_acc))
            time_scheduler = self._cal_time_scheduler(experiment_begin, epoch_idx)
            print(" * Time: {}".format(time_scheduler))

            if acc_mean > best_test_acc:
                best_test_acc = acc_mean
                outfile = os.path.join(self.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch_idx, 'state': self.model.state_dict()}, outfile)

            if (epoch_idx % self.config['save_freq'] == 0) or (epoch_idx == self.config["stop_epoch"] - 1):
                outfile = os.path.join(self.checkpoint_dir, '{:d}.tar'.format(epoch_idx))
                torch.save({'epoch': epoch_idx, 'state':self.model.state_dict()}, outfile)

    def _train(self, epoch_idx):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        self.model.train()
        self.model.train_loop(epoch_idx, self.train_loader, self.optimizer, self.config['print_freq'], self.device)

    def _test(self):
        self.model.eval()
        acc_mean = self.model.test_loop(self.test_loader, self.device)
        return acc_mean
