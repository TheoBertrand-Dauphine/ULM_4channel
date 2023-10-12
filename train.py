import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import ULMDataset, IOSTARDataset
from utils.transforms import RandomCrop, Rescale, ToTensor, HeatMap, RandomAffine, GlobalContrastNormalization, ColorJitter, RandomFlip
from nn.ulm_unet import ULM_UNet, ImagePredictionLogger, Vesselnet

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor

import matplotlib.pyplot as plt
import time

import wandb 
from pytorch_lightning.loggers import WandbLogger

import datetime



def main(args,seed):
    print(args.data)

    if args.data=='IOSTAR':
        data_dir = './data_IOSTAR/'
        train_dataset = IOSTARDataset(root_dir=data_dir + 'train_images', transform=transforms.Compose([RandomCrop(args.size), RandomFlip(), HeatMap(alpha=args.alpha, out_channels = args.out_channels), ToTensor(), RandomAffine(360, 0.1)]), no_endpoints=args.no_endpoints)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        validation_dataset = IOSTARDataset(root_dir=data_dir + 'val_images', transform=transforms.Compose([HeatMap(alpha=args.alpha, out_channels = args.out_channels), ToTensor()]), no_endpoints=args.no_endpoints)
        valloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        if args.data=='synthetic':
            data_dir = './data_synthetic/'
            train_dataset = ULMDataset(root_dir=data_dir + 'train_images', transform=transforms.Compose([RandomCrop(args.size), RandomFlip(), HeatMap(alpha=args.alpha, out_channels = args.out_channels), ToTensor(), RandomAffine(360, 0.1)]))
            trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

            validation_dataset = ULMDataset(root_dir=data_dir + 'val_images', transform=transforms.Compose([HeatMap(alpha=args.alpha, out_channels = args.out_channels), ToTensor()]))
            valloader = DataLoader(validation_dataset, batch_size=21, shuffle=False, num_workers=args.workers)
        else:
            data_dir = './data/'
            train_dataset = ULMDataset(root_dir=data_dir + 'train_images', transform=transforms.Compose([RandomCrop(args.size), GlobalContrastNormalization(), ColorJitter(), RandomFlip(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor(), RandomAffine(360, 0.1)]))
            trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

            validation_dataset = ULMDataset(root_dir=data_dir + 'val_images', transform=transforms.Compose([GlobalContrastNormalization(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor()]))
            valloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=args.workers)

        

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    wandb.login()
    wandb.init(name = "data" + args.data + "_cropsize_" + str(args.size) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + '_NoEndpoints_' + str(args.no_endpoints), config = args)
    wandb_logger = WandbLogger(project="ULM_4CHANNEL")

    if args.data == 'IOSTAR':
        if args.vesselnet:
            model = Vesselnet(in_channels = 3, out_channels = args.out_channels, init_features = args.features, threshold = args.threshold, patience = args.patience, alpha = args.alpha, lr = args.lr, penalization = args.penalization)
        else:
            model = ULM_UNet(in_channels=3, init_features=args.features, threshold = args.threshold, out_channels = args.out_channels, second_unet=args.second_unet, lr = args.lr)
    else:
        model = ULM_UNet(in_channels=1, init_features=args.features, threshold=args.threshold, out_channels = args.out_channels, second_unet=args.second_unet, lr = args.lr)

    samples = next(iter(valloader))

    trainer = Trainer(
        accelerator='gpu',
        devices=-1,
        logger = wandb_logger,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=[ImagePredictionLogger(samples), lr_monitor]
    )

    trainer.fit(model,trainloader,valloader)

    # trainer.save_checkpoint(args.weights + "ulm_net_" + args.data +"_epochs_{}".format(args.epochs) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + "_{}_{}".format(datetime.datetime.today().day, datetime.datetime.today().month) + ".ckpt")

    torch.save(model.state_dict(), args.weights + "ulm_net_" + args.data + "_epochs_{}".format(args.epochs) + "_size_" + str(train_dataset[0]['image'].shape[-1]) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + '_alpha_' + str(args.alpha) + "_{}_{}".format(datetime.datetime.today().day, datetime.datetime.today().month) + '_NoEndpoints_' + str(args.no_endpoints) + ".pt")

    if args.data == 'IOSTAR':
        test_dataset = IOSTARDataset(root_dir=data_dir + 'test_images', transform=transforms.Compose([HeatMap(alpha=args.alpha, out_channels = args.out_channels), ToTensor()]), no_endpoints=args.no_endpoints)

        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
        trainer.test(model, dataloaders = testloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Training U-Net model for segmentation of brain MRI")
    parser.add_argument("--batch-size", type=int, default=10, help="input batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate (default: 0.001)")
    parser.add_argument("--device", type=int, default=1, help="device for training (default: cuda:0)")
    parser.add_argument("--workers",type=int,default=16, help="number of workers for data loading (default: 16)")
    parser.add_argument("--weights", type=str, default="./weights/", help="folder to save weights")
    parser.add_argument("--size",type=int,default=128,help="target input image size (default: 256)")
    parser.add_argument("--data",type=str,default='IOSTAR',help="Using synthetic data (default: ULM data, others : 'synthetic' or 'IOSTAR')")
    parser.add_argument("--patience", type=int, default=200, help=" Number of steps of consecutive stagnation of validation loss before lowering lr (default: 400)")
    parser.add_argument("--threshold", type=float, default=0.05, help="threshold applied on output for detection of points (default: 0.1)")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of channels in the output layer (default: 3)")
    parser.add_argument("--alpha", type=float, default=3., help=" Value of the parameter alpha for gaussian representing landmark (default: 3.)")
    parser.add_argument("--no_endpoints", type=int, default=0, help=" Whether to include endpoints in IOSTAR dataset")
    parser.add_argument("--second_unet", type=int, default=0, help=" Use 2 UNETS?")
    parser.add_argument("--vesselnet", type=int, default=0, help=" Use Vesselnet?")
    parser.add_argument("--features", type=int, default=16, help=" Number of features in first layer")
    parser.add_argument("--penalization", type=float, default=1., help=" Number of features in first layer")


    args = parser.parse_args()

    main(args,42)

