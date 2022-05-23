from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl

import wandb

import torchgeometry

l2loss = nn.MSELoss(reduction='mean')

class ULM_UNet(pl.LightningModule):

    def __init__(self, in_channels=1, out_channels=3, init_features=16, threshold=0.5, patience=400, alpha=1):
        super(ULM_UNet, self).__init__()

        self.threshold = threshold
        self.patience = patience
        self.alpha = alpha

        self.local_max_filt = nn.MaxPool2d(9, stride=1, padding=4)

        features = init_features
        self.encoder1 = ULM_UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ULM_UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ULM_UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ULM_UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ULM_UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ULM_UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ULM_UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ULM_UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ULM_UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
    def forward(self, x):
        # print(x.shape)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience, min_lr=1e-8)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "Normalized_val_loss"}}

    def training_step(self, batch, batch_idx):

        if batch['image'].ndim==4:
            x, y_true = batch['image'], batch['heat_map'].squeeze()
        else:
            x, y_true = batch['image'].unsqueeze(1), batch['heat_map']

        y_pred = self(x)

        # print(y_pred.shape)
        # print(y_true.shape)
        loss = l2loss(y_pred,y_true)
        logs={"train_loss": loss}
        batch_dictionary={
            "loss": loss,
            "log": logs,
        }
        self.log('loss', loss, prog_bar = False, on_step=False,on_epoch=True,logger=True)

        return loss

    def validation_step(self, batch, batch_idx, log=True):

        if batch['image'].ndim==4:
            x, y = batch['image'], batch['heat_map'].squeeze()
        else:
            x, y = batch['image'].unsqueeze(1), batch['heat_map'].squeeze()
        y_hat = self(x)

        # print(y_hat.shape)
        # print(y.shape)
        
        val_loss = l2loss(y_hat,y) #/l2loss(heat_a,torch.zeros_like(heat_a))

        threshold = self.threshold
        dist_tol = 7

        max_output = self.local_max_filt(y_hat)
        detected_points = ((max_output==y_hat)*(y_hat>threshold)).nonzero()

        points_coordinates = batch['landmarks']
        nb_points = ((points_coordinates[:,:,:2]**2).sum(dim=2) > 0).sum(dim=1)

        F1 = torch.tensor(0., device=self.device)
        precision_cum = torch.tensor(0., device=self.device)
        recall_cum = torch.tensor(0., device=self.device)

        avg_points_detected = detected_points.shape[0]/x.shape[0]

        for i in range(x.shape[0]):

            points = detected_points[detected_points[:,0]==i,1:]
            points = points[:,[1,2,0]]

            if (points[:,2]==3).sum()!=0:
                points = points[(points[:,2]!=3),:]

            distance = ((torch.tensor([[[1, 1, 0.*dist_tol]]]).to(device=self.device)*(points.unsqueeze(0) - points_coordinates[i,:nb_points[i],:].unsqueeze(1)))**2).sum(dim=-1)

            if points.shape[0]!=0:
                distance_min = distance*(distance==distance.min(dim=1, keepdim=True).values)
                distance_min[distance!=distance.min(dim=1, keepdim=True).values]=1e8

                found_matrix = (distance_min < dist_tol**2).float()
                
                recall = found_matrix.max(dim=1).values.mean() #nb of points well classified/nb of points in the class
                precision = found_matrix.max(dim=1).values.sum()/max(found_matrix.shape[1],1) #nb of points well classified/nb of points labeled in the class

                recall_cum += recall/x.shape[0]
                precision_cum += precision/x.shape[0]

                if precision!=0 and recall!=0:
                    F1 += 2/((1/recall)+(1/precision))/x.shape[0]

        if log:
            self.log('Normalized_val_loss', val_loss, prog_bar=False, on_step=False,on_epoch=True, logger=True)
            self.log('Precision', precision_cum, prog_bar=False, on_step=False,on_epoch=True, logger=True)
            self.log('Recall', recall_cum, prog_bar=False, on_step=False,on_epoch=True, logger=True)
            self.log('F1 score', F1, prog_bar=False, on_step=False,on_epoch=True, logger=True)
            self.log('Average number of detected_points', avg_points_detected, prog_bar=False, on_step=False,on_epoch=True, logger=True)
        else:
            print(F1)
        return val_loss

def wb_mask(bg_img, pred_mask, true_mask):

    """class_labels = {
        0: 'endpoint',
        1: 'biffurcation',
        2: 'crossing',
    }"""

    return [wandb.Image(bg_img), wandb.Image(pred_mask), wandb.Image(true_mask)]


def gray2rgb(image):
    w, h = image.shape[-2:]
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    if image.ndim==3:
        ret = (255*image).transpose((1,2,0))
    else:
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        if val_samples['image'].ndim==4:
            self.val_imgs, self.val_labels = val_samples['image'], val_samples['heat_map'].squeeze(1)
        else:
            self.val_imgs, self.val_labels = val_samples['image'].unsqueeze(1), val_samples['heat_map'].squeeze(1)

        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)

        mask_list = []

        local_max_filt = nn.MaxPool2d(17, stride=1, padding=8)
        
        for original_image, logits, ground_truth in zip(val_imgs, logits, self.val_labels):
            # the raw background image as a numpy array
            #bg_image = image2np(original_image.data)
            
            bg_image = gray2rgb(original_image.squeeze(0).cpu().numpy()).astype(np.uint8)
            # run the model on that image
            #prediction = pl_module(original_image)[0]

            prediction_mask = np.sum(local_max_filt(logits.clone().detach()).cpu().permute(1,2,0).numpy().astype(np.uint8),axis=-1)

            # ground truth mask
            true_mask = np.sum(local_max_filt(ground_truth.clone().detach()).cpu().permute(1,2,0).numpy().astype(np.uint8),axis=-1)
            # keep a list of composite images
            
            mask_list = wb_mask(bg_image, prediction_mask, true_mask)

        # log all composite images to W&B'''
        wandb.log({"Background" : mask_list[0], "predicted":mask_list[1], "label":mask_list[2]})