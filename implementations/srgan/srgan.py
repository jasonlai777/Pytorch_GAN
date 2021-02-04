"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pprint
from torchvision.utils import save_image, make_grid
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    #print(d_interpolates.shape)
    #print(interpolates.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def Unnormalize(img):
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  unorm0 = img[0,...] * std[0] + mean[0]
  unorm1 = img[1,...] * std[1] + mean[1]
  unorm2 = img[2,...] * std[2] + mean[2]
  unorm = np.dstack((unorm0,unorm1,unorm2))
  return unorm



os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500 , help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=501, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
#parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=1024, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=1024, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
opt = parser.parse_args()

print(opt)

cuda = torch.cuda.is_available()

dataloader = DataLoader(
    ImageDataset("../../data/image/test_img", hr_shape=(1024,1024)),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)      
hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    #generator.load_state_dict(torch.load("saved_model/generator_%d.pth"%opt.epoch))
    #discriminator.load_state_dict(torch.load("saved_model/discriminator_%d.pth"%opt.epoch))
    ###### for test_net_srgan.py
    generator.load_state_dict(torch.load("saved_models_1119_sn1024/generator_%d.pth"%opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models_1119_sn1024/discriminator_%d.pth"%opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor  

# ----------
#  Training
# ----------
img_count = 0
lambda_gp = 10
n_critic = 5
img_loader = []
j=1
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        #print(imgs)
        
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        #hr = imgs_hr[0].detach().cpu().numpy()
        #hr = Unnormalize(hr)
        #hr = hr[::-1,:,:]#.transpose((1,2,0))
        #print(hr.shape)
        #cv2.imwrite("1.jpg", hr*255)
        #exit()
        imgs_names = imgs["name"]
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # Configure input
        #real_imgs = Variable(imgs.type(Tensor))
        '''
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(imgs_lr)

        # Real images
        real_validity = discriminator(imgs_hr)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        #print(real_validity.data, real_validity.data)
        gradient_penalty = compute_gradient_penalty(discriminator, imgs_hr.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(imgs_lr)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()
         
        '''
        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        
        # Adversarial loss
        #loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
        loss_GAN = -math.log(criterion_content(discriminator(gen_hr), valid))

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
        
        
        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] \n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )
        #gen_hr= fake_imgs
        batches_done = epoch * len(dataloader) + i
        
        if epoch == opt.n_epochs-1:
            #torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            #torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
            
            for k in range(gen_hr.shape[0]):
                arr = [imgs["origin_size"][0][k].numpy(), imgs["origin_size"][1][k].numpy()]
                gen_hr_os = F.interpolate(gen_hr, size=(arr[0]*2,arr[1]*2), mode='bilinear') 
                #permute = [2,1,0]
                #gen_hr_os = gen_hr_os[:,permute,:,:]
                
                gen_hr_os_numpy = Unnormalize(gen_hr_os[k].detach().cpu().numpy())
                gen_hr_os_numpy = gen_hr_os_numpy[:,:,::-1]
                print(gen_hr_os_numpy.shape)
                cv2.imwrite("image_final/%s"% imgs_names[k], gen_hr_os_numpy*255)
                #save_image(gen_hr_os_numpy,"image_final/%s" % imgs_names[k],normalize=True)
                print("saved srgan output: %s"% imgs_names[k])
                #img_loader.append(gen_hr_os)

#return img_loader
                
        
        
        #if batches_done % opt.sample_interval == 0:
        '''
        if epoch % 10 == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid_t = torch.cat((imgs_lr, gen_hr), -1)            
            img_grid = Unnormalize(img_grid_t.detach().cpu().numpy())
            img_grid = img_grid[:,:,::-1]
            cv2.imwrite("images/%d.JPG" % (epoch*100+j), img_grid*255)
            #save_image(img_grid, "images/%d.JPG" % (epoch*100+j), normalize=False)
            j+=1
        '''  
    j = 1
          
    '''
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
    '''




