"""General-purpose training script for image-to-image translation.

This script is for training on a dataroot folder with subfolders trainA and trainB. This will train generators and discriminators
A and B and save checkpoints for each epoch as checkpoints in a folder named '--checkpoints_dir'. Note that unlike the original
cycleGAN, more parameters are hard-coded at the end of the file. Primarily, patches are no longer preprocessed by resizing and cropping, and thus,
the preprocessing flag is hard-coded to 'none'.

The main flags you need to specify:
 - '--dataroot'
 - '--checkpoints_dir'
 - '--name' (the name of your model)
 - '--train_mode' (3d or 2d)
 - '--netG'  (the model backbone: e.g. resnet_9block, unet_32, swinunetr)
 - '--patch_size' (The side_length of the patches the dataset is tiled into.
                   If the provided value is not compatible with the backbone, the patch size will be automatically adjusted.)
 - 'stride_A' (the distance between two neighbouring patches for dataset A)
 - 'stride_B' (the distance between two neighbouring patches for dataset B)

Example:
    Train a CycleGAN model:
        python train.py --dataroot path/to/datasets --checkpoints_dir path/to/checkpoints --name my_cyclegan_model
                        --train_mode 3d --netG resnet_9blocks --patch_size 160 --stride_A 160 --stride_B 160

Further parameters such as options for loss, batch_size, epoch number etc. can be seen in the
options/base_options.py and options/train_options.py files.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import gc
import torch
from util.visualizer import Visualizer
from util.util import adjust_patch_size
from argparse import Namespace

def train(opt):
    gc.collect()
    torch.cuda.empty_cache()
    if opt.train_mode == "3d":
        opt.dataset_mode = 'patched_unaligned_3d'
    elif opt.train_mode == "2d":
        opt.dataset_mode = 'patched_unaligned_2d'

    adjust_patch_size(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                #save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(losses) # for the patched dataset I'll use this

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

# def _adjust_patch_size(opt):
#
#     old_patch_size = opt.patch_size
#     if opt.netG.startswith('unet'):
#         depth_factor = int(opt.netG[5:])
#         # print("depth factor: ", depth_factor)
#         patch_size = opt.patch_size
#         # print(patch_size, (patch_size + 2) % depth_factor)
#         if (patch_size + 2) % depth_factor == 0:
#             pass
#         else:
#             # In the valid unet, the patch sizes that can be evenly downsampled in the layers (i.e. without residual) are
#             # limited to values which are divisible by 32 (2**5 for 5 downsampling steps), after adding the pixels lost in the valid conv layer, i.e.:
#             # 158 (instead of 160), 190 (instead of 192), 222 (instead of 224), etc. Below, the nearest available patch size
#             # selected to patch the image accordingly. (Choosing a smaller value than the given patch size, should ensure
#             # that the patches are not bigger than any dimensions of the whole input image)
#             new_patch_sizes = opt.patch_size - torch.arange(1, depth_factor)
#             new_patch_size = int(new_patch_sizes[(new_patch_sizes + 2) % depth_factor == 0])
#             opt.patch_size = new_patch_size
#             print(
#                 f"The provided patch size {old_patch_size} is not compatible with the chosen unet backbone with valid convolutions. Patch size was changed to {new_patch_size}")
#
#     elif opt.netG.startswith("resnet"):
#         patch_size = opt.patch_size
#         if patch_size % 4 == 0:
#             pass
#         else:
#             new_patch_sizes = opt.patch_size - torch.arange(1, 4)
#             new_patch_size = int(new_patch_sizes[(new_patch_sizes % 4) == 0])
#             opt.patch_size = new_patch_size
#             print(
#                 f"The provided patch size {old_patch_size} is not compatible with the resnet backbone. Patch size was changed to {new_patch_size}")
#
#     elif opt.netG.startswith("swinunetr"):
#         patch_size = opt.patch_size
#         if patch_size % 32 == 0:
#             pass
#         else:
#             new_patch_sizes = opt.patch_size - torch.arange(1, 32)
#             new_patch_size = int(new_patch_sizes[(new_patch_sizes % 32) == 0])
#             opt.patch_size = new_patch_size
#             print(
#                 f"The provided patch size {old_patch_size} is not compatible with the swinunetr backbone. Patch size was changed to {new_patch_size}")

if __name__ == '__main__':
    options = {
        'dataroot': '/mnt/md0/rajalakshmi/cycleGAN3D/training_data/',
        'name': 'realrawA_synrawB_upsampled',
        'gpu_ids': [0],
        'checkpoints_dir': '/mnt/md0/rajalakshmi/cycleGAN3D/',
        'model': 'cycle_gan',
        'input_nc': 1,
        'output_nc': 1,
        'ngf': 64,
        'ndf': 64,
        'netD': 'basic',
        'netG': 'unet_32',
        'n_layers_D': 3,
        'norm': 'instance',
        'init_type': 'normal',
        'init_gain': 0.02,
        'no_dropout': True,
        'train_mode': '3d',
        #'dataset_mode': 'patched_unaligned_3d',
        'patch_size': 190,
        'stride_A': 190,
        'stride_B': 190,
        'direction': 'AtoB',
        'phase': 'train',
        'serial_batches': True,
        'num_threads': 32,
        'batch_size': 1,
        'load_size': 128,
        'crop_size': 128,
        'max_dataset_size': float('inf'),
        'preprocess': 'none',
        'no_flip': False,
        'display_winsize': 256,
        'epoch': 'latest',
        'load_iter': 0,
        'verbose': True,
        'suffix': '',
        'use_wandb': True,
        'wandb_project_name': 'CycleGAN3D-master',
        'results_dir': '/mnt/md0/rajalakshmi/cycleGAN3D/generated_dataset/',
        'aspect_ratio': 1.0,
        'n_epochs': 25,
        'n_epochs_decay': 25,
        'beta1': 0.5,
        'lr': 0.0002,
        'gan_mode': 'lsgan',  # vanilla, wganp,
        'pool_size': 50,
        'lr_policy': 'linear',
        'lr_decay_iters': 50,
        'lambda_ssim_G': 0.2,
        'lambda_ssim_cycle': 0.2,
        'display_freq': 10,
        'display_ncols': 3,
        'display_id': 0,
        'display_server': 'localhost',
        'display_env': 'main',
        'display_port': 8097,
        'update_html_freq': 1000,
        'print_freq': 100,
        'no_html': True,
        # network saving and loading parameters
        'save_latest_freq': 2500,
        'save_epoch_freq': 2,
        'save_by_iter': True,
        'continue_train': False,
        'isTrain': True,
        'epoch_count': 1,
        'lambda_A': 10.0,
        'lambda_B': 10.0,
        'lambda_identity': 0.5
    }
    #
    current_options = Namespace(**options)

    train(current_options) # Save the training options in train_opt.txt