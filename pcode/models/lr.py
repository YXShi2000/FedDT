# -*- coding: utf-8 -*-

import torch.nn as nn

__all__ = ["lr"]


class LR(nn.Module):
    def __init__(self, arch, arch_info, dataset, use_hog_feature=False):
        super(LR, self).__init__()
        self.dataset = dataset
        if use_hog_feature:
            self.num_features = 288
        elif dataset == "movie_reviews":
            self.num_features = 2000
        elif "mnist" in dataset:
            self.num_features = 784
        else:
            self.num_features = 1024
        self.num_classes = self._decide_num_classes()
        self.use_hog_feature = use_hog_feature
        self.name = "lr"

        if "lr_s" in arch or "lr_s" in arch_info:
            self.linear = nn.Sequential(
                nn.Linear(self.num_features, self.num_classes),
                nn.Sigmoid()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.num_features, self.num_features // 2),
                nn.ReLU(),
                nn.Linear(self.num_features // 2, self.num_classes),
                nn.Sigmoid(),
            )

        self.activations = None

    def forward(self, x, is_kd=False):
        # convert to grayscale image
        if not self.use_hog_feature and self.dataset not in ["mnist", "emnist", "fashionmnist", "movie_reviews"]:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        if self.dataset in ["mnist", "emnist", "fashionmnist", "movie_reviews"] and is_kd:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        self.activations = x
        return x

    def _decide_num_classes(self):
        if self.dataset in ["cifar10", "svhn", "mnist", "fashionmnist"]:
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "imagenet" in self.dataset:
            return 1000
        elif self.dataset == "femnist":
            return 62
        elif self.dataset == "emnist":
            return 47
        elif self.dataset == "movie_reviews":
            return 4


def lr(conf):
    dataset = conf.data
    model = LR(dataset=dataset, arch=conf.arch, arch_info=conf.complex_arch, use_hog_feature=conf.use_hog_feature)
    return model
