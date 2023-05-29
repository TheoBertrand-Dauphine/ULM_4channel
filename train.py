import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import ULMDataset, IOSTARDataset
from utils.transforms import RandomCrop, Rescale, ToTensor, HeatMap, RandomAffine, GlobalContrastNormalization, ColorJitter, RandomFlip
from nn.ulm_unet import ULM_UNet, ImagePredictionLogger

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
        train_dataset = IOSTARDataset(root_dir=data_dir + 'train_images', transform=transforms.Compose([RandomCrop(args.size), GlobalContrastNormalization(), ColorJitter(), RandomFlip(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor(), RandomAffine(360, 0.1)]), no_endpoints=args.no_endpoints)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        validation_dataset = IOSTARDataset(root_dir=data_dir + 'val_images', transform=transforms.Compose([GlobalContrastNormalization(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor()]), no_endpoints=args.no_endpoints)
        valloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        if args.data=='synthetic':
            data_dir = './data_synthetic/'
            train_dataset = ULMDataset(root_dir=data_dir + 'train_images', transform=transforms.Compose([RandomCrop(args.size), GlobalContrastNormalization(), ColorJitter(), RandomFlip(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor(), RandomAffine(360, 0.1)]))
            trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

            validation_dataset = ULMDataset(root_dir=data_dir + 'val_images', transform=transforms.Compose([GlobalContrastNormalization(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor()]))
            valloader = DataLoader(validation_dataset, batch_size=21, shuffle=False, num_workers=args.workers)
        else:
            data_dir = './data/'
            train_dataset = ULMDataset(root_dir=data_dir + 'train_images', transform=transforms.Compose([RandomCrop(args.size), GlobalContrastNormalization(), ColorJitter(), RandomFlip(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor(), RandomAffine(360, 0.1)]))
            trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

            validation_dataset = ULMDataset(root_dir=data_dir + 'val_images', transform=transforms.Compose([GlobalContrastNormalization(), HeatMap(s=int(3*args.alpha), alpha=args.alpha, out_channels = args.out_channels), ToTensor()]))
            valloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=args.workers)

        

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    wandb.login()
    wandb.init(name = "data" + args.data +"_epochs_" + str(args.epochs) + "_cropsize_" + str(args.size) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + '_alpha_' + str(args.alpha) + '_NoEndpoints_' + str(args.no_endpoints))
    wandb_logger = WandbLogger(project="ULM_4CHANNEL")

    if args.data == 'IOSTAR':
        model = ULM_UNet(in_channels=3, init_features=48, threshold = args.threshold, out_channels = args.out_channels, second_unet=args.second_unet)
    else:
        model = ULM_UNet(in_channels=1, init_features=48, threshold=args.threshold, out_channels = args.out_channels, second_unet=args.second_unet)

    samples = next(iter(valloader))

    trainer = Trainer(
        gpus=args.device,
        #num_nodes=2,
        #accelerator='ddp',
        #plugins=DDPPlugin(find_unused_parameters=False),
        logger = wandb_logger,
        #progress_bar_refresh_rate=0,
        max_epochs=args.epochs,
        #benchmark=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        callbacks=[ImagePredictionLogger(samples), lr_monitor]
    )

    trainer.fit(model,trainloader,valloader)

    # trainer.save_checkpoint(args.weights + "ulm_net_" + args.data +"_epochs_{}".format(args.epochs) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + "_{}_{}".format(datetime.datetime.today().day, datetime.datetime.today().month) + ".ckpt")

    torch.save(model.state_dict(), args.weights + "ulm_net_" + args.data + "_epochs_{}".format(args.epochs) + "_size_" + str(train_dataset[0]['image'].shape[-1]) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + '_alpha_' + str(args.alpha) + "_{}_{}".format(datetime.datetime.today().day, datetime.datetime.today().month) + '_NoEndpoints_' + str(args.no_endpoints) + ".pt")
    # model2 = ULM_UNet(in_channels=3, init_features=48, threshold = args.threshold, out_channels = args.out_channels)
    # # model2.load_from_checkpoint(checkpoint_path = args.weights + "ulm_net_" + args.data +"_epochs_{}".format(args.epochs) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + "_{}_{}".format(datetime.datetime.today().day, datetime.datetime.today().month) + ".ckpt", in_channels=3, init_features=48, threshold = args.threshold, out_channels = args.out_channels)
    # model2.load_state_dict(torch.load(args.weights + "ulm_net_" + args.data +"_epochs_{}".format(args.epochs) + "_batch_{}".format(args.batch_size) + "_out_channels_{}".format(args.out_channels) + "_{}_{}".format(datetime.datetime.today().day, datetime.datetime.today().month) + ".pt"))
    
    # with torch.no_grad():
    #     model.eval()
    #     for val_batch_idx, val_batch in enumerate(valloader):
    #         print(val_batch_idx)
    #         val_out = model.forward(val_batch['image'])
    #         val_out2 = model2.forward(val_batch['image'])

    #         print(((val_out-val_out2)**2).mean())

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Training U-Net model for segmentation of brain MRI")
    parser.add_argument("--batch-size", type=int, default=4, help="input batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate (default: 0.001)")
    parser.add_argument("--device", type=int, default=0, help="device for training (default: cuda:0)")
    parser.add_argument("--workers",type=int,default=1, help="number of workers for data loading (default: 4)")
    parser.add_argument("--weights", type=str, default="./weights/", help="folder to save weights")
    parser.add_argument("--size",type=int,default=512,help="target input image size (default: 256)")
    parser.add_argument("--aug-scale",type=int,default=0.05,help="scale factor range for augmentation (default: 0.05)")
    parser.add_argument("--aug-angle",type=int,default=15,help="rotation angle range in degrees for augmentation (default: 15)")
    parser.add_argument("--data",type=str,default='IOSTAR',help="Using synthetic data (default: ULM data, others : 'synthetic' or 'IOSTAR')")
    parser.add_argument("--patience", type=int, default=400, help=" Number of steps of consecutive stagnation of validation loss before lowering lr (default: 400)")
    parser.add_argument("--threshold", type=float, default=0.5, help="threhsold applied on output for detection of points (default: 0.5)")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of channels in the output layer (default: 3)")
    parser.add_argument("--alpha", type=float, default=3., help=" Value of the parameter alpha for gaussian representing landmark (default: 3.)")
    parser.add_argument("--no_endpoints", type=bool, default=False, help=" Whether to include endpoints in IOSTAR dataset")
    parser.add_argument("--second_unet", type=bool, default=False, help=" Use 2 UNETS?")
    parser.add_argument("--vesselnet", type=bool, default=True, help=" Use Vesselnet?")





    args = parser.parse_args()

    main(args,42)

