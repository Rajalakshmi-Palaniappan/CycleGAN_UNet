
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss, DiceLoss
#from unet_seg_model import *
import torch
from data import create_dataset
from torch.utils.data import DataLoader
from argparse import Namespace
from util.visualizer import Visualizer
from torch.optim.lr_scheduler import LambdaLR
import glob
import tifffile as tiff
from monai.data import Dataset, PatchDataset
import numpy as np

# Set device

def train(opt):

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
     dataset_size = len(dataset)  # get the number of images in the dataset.
     print('The number of training images = %d' % dataset_size)
     print(len(dataset))
     if len(dataset) == 0:
          raise ValueError("Dataset is empty. Please check your data loading implementation.")



     model = UNet(
          spatial_dims=3,
          in_channels=1,
          out_channels=1,
          channels=[16, 32, 64],
          strides=[2, 2],
          norm=Norm.BATCH
     ).to(device)

     loss_function = DiceLoss(sigmoid=True)
     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
     max_epochs = 50

     def linear_decay(epoch):
          return 1 - epoch / max_epochs

     scheduler = LambdaLR(optimizer, lr_lambda=linear_decay)

     visualizer = Visualizer(opt)


     for epoch in range(max_epochs):
          print(f"starting epoch{epoch}")
          model.train()
          epoch_loss = 0.0
          visualizer.reset()
          print("visualizer is reset")
          for batch_idx, (images, labels) in enumerate(dataset):
               images, labels = images.to(device), labels.to(device)

               optimizer.zero_grad()
               outputs = model(images)
               loss = loss_function(outputs, labels)
               loss.backward()
               optimizer.step()
               scheduler.step()


               epoch_loss += loss.item()
               visuals ={'output image': outputs, 'input image': images, 'Labels': labels}
               losses = {'DiceLoss': loss}
               visualizer.display_current_results(visuals, epoch)
               #visualizer.print_current_losses(epoch, loss)
               visualizer.plot_current_losses(losses)

          print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss / len(dataset)}")

     torch.save(model.state_dict(), "unet_segmentation_model.pth")




options = {
     'dataroot': '/mnt/md0/rajalakshmi/cycleGAN3D/training_data_unet/',
     'name': 'UNet_segmentation',
     'gpu_ids': [0],
     'checkpoints_dir': '/mnt/md0/rajalakshmi/cycleGAN3D/UNet_checkpoints',
     #'model': 'unet_segmentation',
    # 'input_nc': 1,
    # 'output_nc': 1,
    # 'ngf': 64,
    # 'ndf': 64,
    # 'netD': 'basic',
     #'netG': 'unet_32',
    # 'n_layers_D': 3,
    # 'norm': 'instance',
    # 'init_type': 'normal',
    # 'init_gain': 0.02,
    # 'no_dropout': True,
     'dataset_mode': 'patched_unaligned_3d',
     'patch_size': 148,
     'stride_A': 148,
     'stride_B': 148,
    # 'direction': 'AtoB',
     'phase': 'train',
     'serial_batches': True,
     'num_threads': 32,
     'batch_size': 1,
     # 'load_size': 190,
     # 'crop_size':96,
     'max_dataset_size': float('inf'),
     'preprocess': 'none',
     #'no_flip': False,
    # 'display_winsize': 256,
    # 'epoch': 'latest',
    # 'load_iter': 0,
    # 'verbose': True,
    # 'suffix': '',
     'use_wandb': True,
     'wandb_project_name': 'UNet_segmentation',
     'train_mode': '3d',
    # 'results_dir': '/mnt/md0/rajalakshmi/cycleGAN3D/generated_dataset/',
    # 'aspect_ratio': 1.0,
    # 'n_epochs': 25,
    # 'n_epochs_decay': 25,
    # 'beta1': 0.5,
    # 'lr': 0.0002,
    # 'gan_mode': 'lsgan',#vanilla, wganp,
    # 'pool_size': 50,
    # 'lr_policy': 'linear',
    # 'lr_decay_iters': 50,
    # 'lambda_ssim_G': 0.2,
    # 'lambda_ssim_cycle': 0.2,
    'display_freq': 100,
    # 'display_ncols': 3,
    # 'display_id': 0,
    # 'display_server': 'localhost',
    # 'display_env': 'main',
    # 'display_port': 8097,
    # 'update_html_freq': 1000,
    # 'print_freq': 100,
    # 'no_html': True,
    # # network saving and loading parameters
    # 'save_latest_freq': 2500,
    # 'save_epoch_freq': 2,
    # 'save_by_iter': True,
    # 'continue_train': False,
     'isTrain': True,
    # 'epoch_count': 1,
    # 'lambda_A': 10.0,
    # 'lambda_B': 10.0,
    # 'lambda_identity': 0.5
}
#
current_options = Namespace(**options)

train(current_options)


