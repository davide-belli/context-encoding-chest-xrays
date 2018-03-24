from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import pickle
import math
import time

from model import _netjointD, _netlocalD, _netG
from utils import plotter, generate_directories
from psnr import psnr

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--test_workers', type=int, help='number of data loading workers on testset', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the patch to be reconstructed')
parser.add_argument('--initialScaleTo', type=int, default=1024,
                    help='the height / width to rescale the original image before eventual cropping')
parser.add_argument('--name', default='exp', help='the name of the experiment used for the directory')

# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=1)  # By default convert image input and output to grayscale
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--update_train_img', type=int, default=100, help='how often (iterations) to update training set images')
parser.add_argument('--update_measures_plots', type=int, default=200, help='how often (iterations) to add a new datapoint in measure plots')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--freeze_disc', type=int, default=1, help='every how many iterations do improvement step on Disc')
parser.add_argument('--freeze_gen', type=int, default=1, help='every how many iterations do improvement step on Gen')
parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')
parser.add_argument('--continueTraining', action='store_true', help='Continue Training from existing model checkpoint')

# parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
parser.add_argument('--wtl2', type=float, default=0.998, help='0 means do not use else use with this weight')

parser.add_argument('--randomCrop', action='store_true', help='Run experiment with RandomCrop')
parser.add_argument('--N_randomCrop', type=int, default=10)
parser.add_argument('--PAD_randomCrop', type=int, default=0)
parser.add_argument('--CENTER_SIZE_randomCrop', type=int, default=768)

parser.add_argument('--jointD', action='store_true', help='Discriminator joins Local and Global Discriminators')
parser.add_argument('--fullyconn_size', type=int, default=1024, help='Size of the output of Local and Global Discriminator which will be joint in fully conntected layer')
parser.add_argument('--patch_with_margin_size', type=int, default=80, help='the size of image with margin to extend the reconstructed center to be input in Local Discriminator')

opt = parser.parse_args()
opt.cuda = True

# opt.ndf = 128 #Discriminator
# opt.nef = 128 #Generator
# opt.imageSize = 128
# opt.patchSize = 64
# opt.initialScaleTo = 128
# opt.patch_with_margin_size = 64

# opt.freeze_gen = 1
# opt.freeze_disc = 5
# opt.jointD = True
# opt.randomCrop = True
# opt.continueTraining = False
# opt.name = "MARGIN_exp"
# opt.fullyconn_size = 512
# opt.update_train_img = 50
# opt.update_measures_plots = 50

# Path parameters
if opt.randomCrop:
    opt.name += "_randomCrop"
if opt.jointD:
    opt.name += "_jointD"
EXP_NAME = "".join(
    [opt.name, '_imageSize', str(opt.imageSize), '_patchSize', str(opt.patchSize), '_nef', str(opt.nef), '_ndf',
     str(opt.ndf)])


print(opt)
print("\nStarting Experiment:", EXP_NAME, "\n")

if opt.continueTraining:
    print("Continuing with the training of the existing model in:", "./outputs/" + EXP_NAME)

# Create dictionary of directory paths
PATHS = dict()
PATHS["netG"] = "outputs/" + EXP_NAME + "/netG_context_encoder.pth"
PATHS["netD"] = "outputs/" + EXP_NAME + "/netD_discriminator.pth"
PATHS["measures"] = "outputs/" + EXP_NAME + "/measures.pickle"
PATHS["train"] = "outputs/" + EXP_NAME + "/train_results"
PATHS["test"] = "outputs/" + EXP_NAME + "/test_results"
PATHS["plots"] = "outputs/" + EXP_NAME + "/plots"
PATHS["randomCrops"] = "outputs/" + EXP_NAME + "/test_results/randomCrops"

generate_directories(PATHS, EXP_NAME, opt.randomCrop)

# Seeds
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.randomCrop:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(opt.initialScaleTo),
        transforms.CenterCrop(opt.CENTER_SIZE_randomCrop),
        transforms.RandomCrop(opt.imageSize, opt.PAD_randomCrop),
        transforms.ToTensor(),
    ])
    transform_original = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(opt.initialScaleTo),
        transforms.ToTensor(),
    ])
    datasets = []
    test_datasets = []
    test_original = []
    for i in range(opt.N_randomCrop):
        datasets.append(dset.ImageFolder(root='dataset_lungs/train', transform=transform))
        test_datasets.append(dset.ImageFolder(root='dataset_lungs/test_64', transform=transform))
    dataset = torch.utils.data.ConcatDataset(datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    test_original = dset.ImageFolder(root='dataset_lungs/test_64', transform=transform_original)
    test_original_dataloader = torch.utils.data.DataLoader(test_original, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.test_workers))

else:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(opt.initialScaleTo),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
    ])
    dataset = dset.ImageFolder(root='dataset_lungs/train', transform=transform)
    test_dataset = dset.ImageFolder(root='dataset_lungs/test_64', transform=transform)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
assert test_dataset
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.test_workers))

wtl2 = float(opt.wtl2)
overlapL2Weight = 10


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_image(image, epoch, path_to_save, name):
    vutils.save_image(image,
                      path_to_save + '/epoch_%03d_' % epoch + name + '.png')


resume_epoch = 0

netG = _netG(opt)
netG.apply(weights_init)
if opt.continueTraining:
    print("Loading model netG from: ", PATHS["netG"])
    netG.load_state_dict(torch.load(PATHS["netG"], map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(PATHS["netG"])['epoch']
print(netG)

if opt.jointD:
    netD = _netjointD(opt)
else:
    netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.continueTraining:
    print("Loading model netD from: ", PATHS["netD"])
    netD.load_state_dict(torch.load(PATHS["netD"], map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(PATHS["netD"])['epoch']
print(netD)

print("\n")

if opt.continueTraining:
    print("Contuining from resume epoch:", resume_epoch)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(opt.batchSize, 1, int(opt.patchSize), int(opt.patchSize))
real_center_plus_margin = torch.FloatTensor(opt.batchSize, 1, int(opt.patch_with_margin_size), int(opt.patch_with_margin_size))
fake_center_plus_margin = torch.FloatTensor(opt.batchSize, 1, int(opt.patch_with_margin_size), int(opt.patch_with_margin_size))


print("Moving models to CUDA...")

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()
    real_center_plus_margin = real_center_plus_margin.cuda()
    fake_center_plus_margin = fake_center_plus_margin.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

real_center = Variable(real_center)
real_center_plus_margin = Variable(real_center_plus_margin)
fake_center_plus_margin = Variable(fake_center_plus_margin)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

D_G_zs = []
D_xs = []
Advs = []
L2s = []
D_tots = []
G_tots = []

# Load measures from initial part of the training, if loading an existing model
if opt.continueTraining:
    (D_G_zs, D_xs, Advs, L2s, G_tots, D_tots) = pickle.load(open(PATHS["measures"], "rb"))
    print("Loaded saved measures with ", len(D_G_zs), "datapoints, approximately ",
          math.ceil(len(D_G_zs) * opt.update_measures_plots / len(dataloader)), "epochs")

this_DGz = 0
this_Dx = 0
this_Adv = 0
this_L2 = 0
this_G_tot = 0
this_D_tot = 0

print("Starting training in 5s...")
time.sleep(5)

step_counter = 0

for epoch in range(resume_epoch, opt.niter):
    epoch_time = time.time()

    #################################
    # Training part for every epoch #
    #################################
    for i, data in enumerate(dataloader, 0):
        step_counter += 1
        
        real_cpu, _ = data
        real_center_cpu = real_cpu[:, :,
                          int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2),
                          int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped.data[:, 0,
        int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
            opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
        int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
            opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
        if opt.nc > 1:
            input_cropped.data[:, 1,
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
            input_cropped.data[:, 2,
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0
        
        # print(type(input_real), type(input_real.data), type(input_cropped), type(real_cpu), type(real_center_cpu), type(real_center), type(label))
        # train with real
        netD.zero_grad()
        label.data.resize_(batch_size, 1).fill_(real_label)
        
        # input("Proceed..." + str(real_center.data.size()))
        
        # print(real_center.data.size(), input_real.data.size())
        if opt.jointD:
            if opt.patchSize != opt.patch_with_margin_size:
                real_center_plus_margin.data.copy_(real_cpu[:, :,
                    int(opt.imageSize / 2 - opt.patch_with_margin_size / 2):int(opt.imageSize / 2 + opt.patch_with_margin_size / 2),
                    int(opt.imageSize / 2 - opt.patch_with_margin_size / 2):int(opt.imageSize / 2 + opt.patch_with_margin_size / 2)])
                output = netD(real_center_plus_margin, input_real)
            else:
                output = netD(real_center, input_real)
        else:
            output = netD(real_center)
        # print(output.data.size())
        # print(label.data.size())
        # input()
        errD_real = criterion(output, label)
        if i % opt.freeze_disc == 0:  # Step Discriminator every freeze_gen iterations
            errD_real.backward()
        D_x = output.data.mean()
        
        # train with fake

        # print(input_cropped.size())
        fake = netG(input_cropped)
        # print(type(fake)) #Variable
        # print(fake.data.size(), " ", input_cropped.data.size())
        label.data.fill_(fake_label)
        if opt.jointD:
            recon_image = input_cropped.clone()
            recon_image.data[:, :,
            int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2),
            int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)] = fake.data

            if opt.patchSize != opt.patch_with_margin_size:
                fake_center_plus_margin.data.copy_(recon_image.data[:, :,
                                                   int(opt.imageSize / 2 - opt.patch_with_margin_size / 2):int(
                                                       opt.imageSize / 2 + opt.patch_with_margin_size / 2),
                                                   int(opt.imageSize / 2 - opt.patch_with_margin_size / 2):int(
                                                       opt.imageSize / 2 + opt.patch_with_margin_size / 2)])
                output = netD(fake_center_plus_margin.detach(), recon_image.detach())
            else:
                output = netD(fake.detach(), recon_image.detach())
        else:
            output = netD(fake.detach())
        # print(output.data.size(), " ", label.data.size())
        # input("")
        errD_fake = criterion(output, label)
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake

        if i % opt.freeze_disc == 0:  # Step Discriminator every freeze_gen iterations
            errD_fake.backward()
            optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        if opt.jointD:
            if opt.patchSize != opt.patch_with_margin_size:
                output = netD(fake_center_plus_margin, recon_image)
            else:
                output = netD(fake, recon_image)
        else:
            output = netD(fake)
        errG_D = criterion(output, label)
        
        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2 * overlapL2Weight)
        wtl2Matrix.data[:, :, int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred),
        int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred)] = wtl2
        
        errG_l2 = (fake - real_center).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()
        
        errG = (1 - wtl2) * errG_D + wtl2 * errG_l2
        
        if i % opt.freeze_gen == 0:  # Step Generator every freeze_gen iterations
            errG.backward()
            
            D_G_z2 = output.data.mean()
            optimizerG.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f | Loss_G (Adv/L2->Tot): %.4f / %.4f -> %.4f | p_D(x): %.4f | p_D(G(z)): %.4f'
              % (epoch + 1, opt.niter, i + 1, len(dataloader),
                 errD.data[0], errG_D.data[0] * (1 - wtl2), errG_l2.data[0] * wtl2, errG.data[0], D_x, D_G_z1))
        
        this_DGz += D_G_z1
        this_Dx += D_x
        this_Adv += errG_D.data[0]
        this_L2 += errG_l2.data[0]
        this_G_tot += errG.data[0]
        this_D_tot += errD.data[0]
        
        if step_counter == opt.update_measures_plots:
            this_Adv *= (1 - wtl2)
            this_L2 *= wtl2
            this_DGz /= opt.update_measures_plots
            this_Dx /= opt.update_measures_plots
            this_Adv /= opt.update_measures_plots
            this_L2 /= opt.update_measures_plots
            this_G_tot /= opt.update_measures_plots
            this_D_tot /= opt.update_measures_plots
            
            D_G_zs.append(this_DGz)
            D_xs.append(this_Dx)
            Advs.append(this_Adv)
            L2s.append(this_L2)
            G_tots.append(this_G_tot)
            D_tots.append(this_D_tot)
            
            plotter(D_G_zs, D_xs, Advs, L2s, G_tots, D_tots, (len(dataloader) / opt.update_measures_plots), PATHS["plots"])
            
            this_DGz = 0
            this_Dx = 0
            this_Adv = 0
            this_L2 = 0
            this_G_tot = 0
            this_D_tot = 0
            step_counter = 0
        
        if i % opt.update_train_img == 0:
            if not opt.jointD:
                recon_image = input_cropped.clone()
                recon_image.data[:, :,
                int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2),
                int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)] = fake.data
            save_image(real_cpu, epoch+1, PATHS["train"], "real")
            save_image(recon_image.data, epoch + 1, PATHS["train"], "recon")
            save_image(fake_center_plus_margin.data, epoch + 1, PATHS["train"], "center_recon")
            save_image(real_center_plus_margin.data, epoch + 1, PATHS["train"], "center_real")
    
    
    
    
    #####################################
    # Testing at the end of every epoch #
    #####################################
    
    if epoch == 0:
        # Clear existing file if any
        with open(PATHS["test"] + "/PSNRs.txt", "w") as myfile:
            myfile.write("")
            
    with open(PATHS["test"] + "/PSNRs.txt", "a") as myfile:
        myfile.write("\nEPOCH " + str(epoch))
        
    for i, data in enumerate(test_dataloader, 0):
        real_cpu, _ = data
        real_center_cpu = real_cpu[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2),
                          int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2)]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped.data[:, 0,
        int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
            opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
        int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
            opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
        if opt.nc > 1:
            input_cropped.data[:, 1,
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
            input_cropped.data[:, 2,
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
            int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0

        fake = netG(input_cropped)
        recon_image = input_cropped.clone()
        recon_image.data[:, :,
        int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2),
        int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)] = fake.data
        
        real_center_np = (real_center.data.cpu().numpy() + 1) * 127.5
        fake_np = (fake.data.cpu().numpy() + 1) * 127.5
        real_cpu_np = (real_cpu.cpu().numpy() + 1) * 127.5
        recon_image_np = (recon_image.data.cpu().numpy() + 1) * 127.5
        
        # Compute PSNR
        p = 0
        total_p = 0
        for j in range(opt.batchSize):
            p += psnr(real_center_np[j].transpose(1, 2, 0), fake_np[j].transpose(1, 2, 0))
            total_p += psnr(real_cpu_np[j].transpose(1, 2, 0), recon_image_np[j].transpose(1, 2, 0))
        
        print('[%d/%d] PSNR per Patch: %.4f | PSNR per Image: %.4f'
              % (i + 1, len(test_dataloader), p / opt.batchSize, total_p / opt.batchSize))

        with open(PATHS["test"] + "/PSNRs.txt", "a") as myfile:
            myfile.write('\n\t[%d/%d] PSNR per Patch: %.4f | PSNR per Image: %.4f'
              % (i + 1, len(test_dataloader), p / opt.batchSize, total_p / opt.batchSize))
        
        save_image(real_cpu, epoch+1, PATHS["test"], "_"+str(i)+"real")
        save_image(recon_image.data, epoch + 1, PATHS["test"], "_"+str(i)+"recon")
        
    if opt.randomCrop:
        MIN = (opt.initialScaleTo - opt.CENTER_SIZE_randomCrop) // 2
        MAX = (opt.initialScaleTo + opt.CENTER_SIZE_randomCrop) // 2 - opt.imageSize
        input_rc_cropped = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
        input_rc_cropped = Variable(input_rc_cropped.cuda())
        for data in test_original_dataloader:
            for k in range(2):
                image_1024 = data[0][k].view(-1, 1, opt.initialScaleTo, opt.initialScaleTo)
                image_1024_recon = image_1024.clone()
                for l in range(3):
                    x = random.randint(MIN, MAX)
                    y = random.randint(MIN, MAX)
                    # print(x, y)
                    real_cpu = image_1024[:, :, x:(x + opt.imageSize), y:(y + opt.imageSize)]
                    input_rc_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
                    input_rc_cropped.data[:, 0,
                    int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                        opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred),
                    int(opt.imageSize / 2 - opt.patchSize / 2 + opt.overlapPred):int(
                        opt.imageSize / 2 + opt.patchSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
                    
                    fake = netG(input_rc_cropped.expand(opt.batchSize, -1, -1, -1))
                    recon_image = input_rc_cropped.clone()
                    recon_image.data[:, :,
                    int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2),
                    int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)] = fake.data[0].view(-1, 1, opt.patchSize, opt.patchSize) # TODO check what fake is and why reshaping
                    save_image(real_cpu, epoch+1, PATHS["randomCrops"], str(k) + "_sub_"+str(l)+"real")
                    save_image(recon_image.data, epoch + 1, PATHS["randomCrops"],
                               str(k) + "_sub_" + str(l)+"recon")
                    save_image(input_rc_cropped.expand(opt.batchSize, -1, -1, -1).data, epoch+1, PATHS["randomCrops"], str(k) + "_batches_" + str(l)+"recon")
                    save_image( fake.data, epoch + 1, PATHS["randomCrops"], str(k) + "_batches_" + str(l)+"fake")
                    
                    image_1024_recon[:, :, int(x + opt.imageSize/2 - opt.patchSize/2):int(x + opt.imageSize/2 + opt.patchSize/2), int(y + opt.imageSize/2 - opt.patchSize/2):int(y + opt.imageSize/2 + opt.patchSize/2)] = fake.data[0].view(-1, 1, opt.patchSize, opt.patchSize)
                save_image(image_1024, epoch+1, PATHS["randomCrops"], str(k) + "_"+"real")
                save_image(image_1024_recon, epoch + 1, PATHS["randomCrops"], str(k) + "_"+"recon")

    # Store model checkpoint
    torch.save({'epoch': epoch + 1, 'state_dict': netG.state_dict()}, PATHS["netG"])
    torch.save({'epoch': epoch + 1, 'state_dict': netD.state_dict()}, PATHS["netD"])

    # Store measure lists
    pickle.dump((D_G_zs, D_xs, Advs, L2s, G_tots, D_tots), open(PATHS["measures"], "wb"))
    
    print("\tEpoch", epoch + 1, "took ", (time.time() - epoch_time) / 60, "minutes")
