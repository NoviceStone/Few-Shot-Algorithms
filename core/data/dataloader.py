# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from core.data.dataset import MetaMSTAR


def get_dataloader(config, mode):
    """Get the dataloader corresponding to the model type and training phase.

    According to the config dict, the training phase and model category, select the appropriate transforms, set the corresponding sampler and collate_fn, and return the corresponding dataloader.

    Args:
        config (dict): A LibFewShot setting dict
        mode (str): mode in train/test/val
        model_type (ModelType): model type in meta/metric//finetuning

    Returns:
        Dataloader: The corresponding dataloader.
    """

    trfms_list = []

    """
    Attention that: Conv and ResNet have different input image resolution
    """
    # Add user's trfms here (or in get_augment_method())
    if 'Conv' in config['model']:
        trfms_list.append(transforms.Resize((96, 96)))
        trfms_list.append(transforms.CenterCrop((84, 84)))
        trfms_list.append(transforms.ToTensor())
        trfms = transforms.Compose(trfms_list)
    elif 'ResNet' in config['model']:
        trfms_list.append(transforms.Resize((96, 96)))
        trfms_list.append(transforms.CenterCrop((84, 84)))
        trfms_list.append(transforms.Resize((224, 224)))
        trfms_list.append(transforms.ToTensor())
        trfms = transforms.Compose(trfms_list)
    else:
        raise  ValueError('Unknown backbone module, please define by yourself')

    dataset = MetaMSTAR(
        args=config,
        root=config["data_root"],
        train=mode == 'train',
        transform=trfms,
        method_type=config['type']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["n_gpu"] * 0,  # for some magic reason, num_workers have to be 0
        drop_last=True,
        pin_memory=True,
    )

    return dataloader
