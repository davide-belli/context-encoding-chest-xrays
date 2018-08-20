from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
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
import numpy as np
from os import listdir

from model import _netjointD, _netlocalD, _netG, _netmarginD
from utils import plotter, generate_directories, psnr

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
parser.add_argument('--update_train_img', type=int, default=10000, help='how often (iterations) to update training set images')
parser.add_argument('--update_measures_plots', type=int, default=200, help='how often (iterations) to add a new datapoint in measure plots')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--freeze_disc', type=int, default=1, help='every how many iterations do improvement step on Disc')
parser.add_argument('--freeze_gen', type=int, default=1, help='every how many iterations do improvement step on Gen')
parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')
parser.add_argument('--continueTraining', action='store_true', help='Continue Training from existing model checkpoint')
parser.add_argument('--register_hooks', action='store_true', help='Add Hooks to debug gradients in padding during training')
parser.add_argument('--inpaintTest', action='store_true', help='Inpaint back test patches on original images when testing')

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
parser.add_argument('--marginD', action='store_true', help='Discriminator with margins')
parser.add_argument('--freezeTraining', action='store_true', help='2 Epochs Gen, 5 Epochs Disc, then combined')

parser.add_argument('--output', default='reconstructions/', help='the name of the experiment used for testing')

opt = parser.parse_args()
opt.cuda = True

# opt.ndf = 128 #Discriminator
# opt.nef = 128 #Generator
# opt.imageSize = 128
# opt.patchSize = 64
# opt.initialScaleTo = 128
# opt.patch_with_margin_size = 66 # 80

# opt.freeze_gen = 1
# opt.freeze_disc = 5
opt.continueTraining = True
# opt.jointD = True
# opt.marginD = True
# opt.randomCrop = True
# opt.name = "TO BE DELETED"
# opt.fullyconn_size = 512
# opt.update_train_img = 200
# opt.wtl2 = 0
# opt.register_hooks = True
# opt.freezeTraining = True

LIMIT_TRAINING = 1000000
# LIMIT_TRAINING = 1000

# torch.set_printoptions(threshold=5000)

# Path parameters
if opt.randomCrop:
    opt.name += "_randomCrop"
if opt.jointD:
    opt.name += "_jointD"
elif opt.marginD:
    opt.name += "_marginD"
EXP_NAME = "".join(
    [opt.name, '_imageSize', str(opt.imageSize), '_patchSize', str(opt.patchSize), '_nef', str(opt.nef), '_ndf',
     str(opt.ndf)])

print(opt)
print("\nStarting Experiment:", EXP_NAME, "\n")


opt.output += EXP_NAME + "/"

TEST_HEALTHY = "dataset_lungs/healthy880patch"
TEST_UNHEALTHY = "dataset_lungs/unhealthy880patch"
TEST_PATCHES = "dataset_lungs/patches"
BASE_HEALTHY = "healthy880patch/"
BASE_UNHEALTHY = "unhealthy880patch/"
BASE_PATCHES = "patches/"
OUT_HEALTHY = opt.output + BASE_HEALTHY
OUT_UNHEALTHY = opt.output + BASE_UNHEALTHY
OUT_PATCHES = opt.output + BASE_PATCHES


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

# generate_directories(PATHS, EXP_NAME, opt.randomCrop)

try:
    os.makedirs(opt.output)
except OSError:
    pass

try:
    os.makedirs(OUT_HEALTHY)
except OSError:
    pass

try:
    os.makedirs(OUT_UNHEALTHY)
except OSError:
    pass

try:
    os.makedirs(OUT_PATCHES)
except OSError:
    pass



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
    transform_randomPatches = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    # datasets = []
    # test_datasets = []
    # test_original = []
    # for i in range(opt.N_randomCrop):
        # datasets.append(dset.ImageFolder(root='dataset_lungs/train', transform=transform))
        # test_datasets.append(dset.ImageFolder(root='dataset_lungs/test_64', transform=transform))
    # dataset = torch.utils.data.ConcatDataset(datasets)
    # test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    
    # dataset = dset.ImageFolder(root='dataset_lungs/train_randomPatches', transform=transform_randomPatches)
    test_healthy = dset.ImageFolder(root=TEST_HEALTHY, transform=transform_randomPatches)
    test_unhealthy = dset.ImageFolder(root=TEST_UNHEALTHY, transform=transform_randomPatches)
    test_patches = dset.ImageFolder(root=TEST_PATCHES, transform=transform_randomPatches)
    
    # test_original = dset.ImageFolder(root='dataset_lungs/test_64', transform=transform_original)
    # test_original_dataloader = torch.utils.data.DataLoader(test_original, batch_size=opt.batchSize,
    #                                               shuffle=False, num_workers=int(opt.test_workers))

else:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(opt.initialScaleTo),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
    ])
    dataset = dset.ImageFolder(root='dataset_lungs/train', transform=transform)
    test_dataset = dset.ImageFolder(root='dataset_lungs/test_64', transform=transform)

# assert dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))
assert test_healthy
healthy_dataloader = torch.utils.data.DataLoader(test_healthy, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.test_workers))

assert test_unhealthy
unhealthy_dataloader = torch.utils.data.DataLoader(test_unhealthy, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.test_workers))

assert test_patches
patches_dataloader = torch.utils.data.DataLoader(test_patches, batch_size=opt.batchSize,
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

def save_single_image(image, path_to_save, name):
    vutils.save_image(image,
                      path_to_save + name + '.png')
    
# def save_grad(image):
#     # print("original")
#     print(image.data)
#     print("max value", torch.max(image.data))
#     image.data = image.data/torch.max(image.data)
#     # print(image.data)
#     vutils.save_image(image.data,
#                       PATHS["train"] + '/epoch_%03d_' % epoch +  str(time.time()) +'.png')

def recursive_image_finder(path, images_list):
    file_list = listdir(path)
    for f in file_list:
        if ".png" not in f:
            images_list = recursive_image_finder(path + "/" + f, images_list)
        else:
            images_list.append((f.replace(".png", "")))
    
    return images_list

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
elif opt.marginD:
    netD = _netmarginD(opt)
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

paddingLayerWhole = nn.ZeroPad2d((opt.imageSize - opt.patchSize)//2)
paddingLayerMargin = nn.ZeroPad2d((opt.patch_with_margin_size - opt.patchSize)//2)
# paddingLayerMargin_real = nn.ZeroPad2d((opt.patch_with_margin_size - opt.patchSize)//2)
# paddingMargin = ((opt.patch_with_margin_size - opt.patchSize)//2, (opt.patch_with_margin_size - opt.patchSize)//2, (opt.patch_with_margin_size - opt.patchSize)//2, (opt.patch_with_margin_size - opt.patchSize)//2)
# paddingWhole = ((opt.imageSize - opt.patchSize)//2, (opt.imageSize - opt.patchSize)//2, (opt.imageSize - opt.patchSize)//2, (opt.imageSize - opt.patchSize)//2)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(opt.batchSize, 1, int(opt.patchSize), int(opt.patchSize))
real_center_plus_margin = torch.FloatTensor(opt.batchSize, 1, int(opt.patch_with_margin_size), int(opt.patch_with_margin_size))


print("Moving models to CUDA...")

if opt.cuda:
    netD.cuda()
    netG.cuda()
    paddingLayerMargin.cuda()
    # paddingLayerMargin_real.cuda()
    paddingLayerWhole.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()
    real_center_plus_margin = real_center_plus_margin.cuda()
    
if opt.randomCrop:
    image_1024 = torch.FloatTensor(opt.batchSize, 1, opt.initialScaleTo, opt.initialScaleTo)
    image_1024 = Variable(image_1024.cuda())
    input_rc_cropped = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
    input_rc_cropped = Variable(input_rc_cropped.cuda())

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

real_center = Variable(real_center)
real_center_plus_margin = Variable(real_center_plus_margin)

netG.eval()
netD.eval()

print("Starting testing in 3s...")
time.sleep(3)

    
    #####################################
    # Testing at the end of every epoch #
    #####################################
    
PSNR_TOTAL = opt.output + "/TOTAL_PSNRs.txt"
PSNR_HEALTHY = opt.output + "/HEALTHY_PSNRs.txt"
PSNR_UNHEALTHY = opt.output + "/UNHEALTHY_PSNRs.txt"
PSNR_PATCHES = opt.output + "/PATCHES_PSNRs.txt"


names_healthy = recursive_image_finder(TEST_HEALTHY, [])
names_unhealthy = recursive_image_finder(TEST_UNHEALTHY, [])
names_patches = recursive_image_finder(TEST_PATCHES, [])
names_healthy.sort()
names_unhealthy.sort()
names_patches.sort()
# print(len(names_patches), len(names_healthy), len(names_unhealthy))


with open(PSNR_TOTAL, "w") as myfile:
    myfile.write("")
    
with open(PSNR_HEALTHY, "w") as myfile:
    myfile.write("")
    
with open(PSNR_UNHEALTHY, "w") as myfile:
    myfile.write("")

with open(PSNR_PATCHES, "w") as myfile:
    myfile.write("")
    
# HEALTHY TEST
print("Testing healthy images...")
tot_psnr_patch_healthy = []
tot_psnr_image_healthy = []

for i, data in enumerate(healthy_dataloader, 0):
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
    
    # print(real_center.data[0,0,0:2,:])
    # input()
    # print(np.amin(real_center.data.cpu().numpy()), np.amax(real_center.data.cpu().numpy()))
    real_center_np = (real_center.data.cpu().numpy()) * 255
    fake_np = (fake.data.cpu().numpy()) * 255
    real_cpu_np = (real_cpu.cpu().numpy()) * 255
    recon_image_np = (recon_image.data.cpu().numpy())* 255
    # print(real_center_np[0,0,0:2,:])
    # input()
    
    # Compute PSNR
    p = 0
    total_p = 0
    for j in range(real_center_np.shape[0]):
        this_p = psnr(real_center_np[j].transpose(1, 2, 0), fake_np[j].transpose(1, 2, 0))
        this_total_p = psnr(real_cpu_np[j].transpose(1, 2, 0), recon_image_np[j].transpose(1, 2, 0))
        with open(PSNR_HEALTHY, "a") as myfile:
            myfile.write('\n\t[Image %d] PSNR per Patch: %.4f | PSNR per Image: %.4f'
                  % (j + opt.batchSize*i, this_p, this_total_p))
        # print('[Image %d] PSNR per Patch: %.4f | PSNR per Image: %.4f' % (j + opt.batchSize*i, this_p, this_total_p))
        save_single_image(real_cpu[j], OUT_HEALTHY, names_healthy[j + opt.batchSize*i] + "_" + "real")
        save_single_image(recon_image.data[j], OUT_HEALTHY, names_healthy[j + opt.batchSize*i] + "_" + "recon")
        
        tot_psnr_patch_healthy.append(this_p)
        tot_psnr_image_healthy.append(this_total_p)
    
    


# UNHEALTHY TEST
print("Testing unhealthy images...")

tot_psnr_patch_unhealthy = []
tot_psnr_image_unhealthy = []

for i, data in enumerate(unhealthy_dataloader, 0):
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

    real_center_np = (real_center.data.cpu().numpy()) * 255
    fake_np = (fake.data.cpu().numpy()) * 255
    real_cpu_np = (real_cpu.cpu().numpy()) * 255
    recon_image_np = (recon_image.data.cpu().numpy()) * 255

    # Compute PSNR
    p = 0
    total_p = 0
    for j in range(real_center_np.shape[0]):
        this_p = psnr(real_center_np[j].transpose(1, 2, 0), fake_np[j].transpose(1, 2, 0))
        this_total_p = psnr(real_cpu_np[j].transpose(1, 2, 0), recon_image_np[j].transpose(1, 2, 0))
        with open(PSNR_UNHEALTHY, "a") as myfile:
            myfile.write('\n\t[Image %d] PSNR per Patch: %.4f | PSNR per Image: %.4f'
                         % (j + opt.batchSize*i, this_p, this_total_p))
        # print('[Image %d] PSNR per Patch: %.4f | PSNR per Image: %.4f' % ((j + opt.batchSize*i), this_p, this_total_p))
        save_single_image(real_cpu[j], OUT_UNHEALTHY, names_unhealthy[j + opt.batchSize*i] + "_" + "real")
        save_single_image(recon_image.data[j], OUT_UNHEALTHY, names_unhealthy[j + opt.batchSize*i] + "_" + "recon")

        tot_psnr_patch_unhealthy.append(this_p)
        tot_psnr_image_unhealthy.append(this_total_p)




# PATCHES TEST
print("Testing patches images...")
tot_psnr_patch_patches = []
tot_psnr_image_patches = []

for i, data in enumerate(patches_dataloader, 0):
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
    
    # print(real_center.data[0,0,0:2,:])
    # input()
    # print(np.amin(real_center.data.cpu().numpy()), np.amax(real_center.data.cpu().numpy()))
    real_center_np = (real_center.data.cpu().numpy()) * 255
    fake_np = (fake.data.cpu().numpy()) * 255
    real_cpu_np = (real_cpu.cpu().numpy()) * 255
    recon_image_np = (recon_image.data.cpu().numpy()) * 255
    # print(real_center_np[0,0,0:2,:])
    # input()
    
    # Compute PSNR
    p = 0
    total_p = 0
    for j in range(real_center_np.shape[0]):
        this_p = psnr(real_center_np[j].transpose(1, 2, 0), fake_np[j].transpose(1, 2, 0))
        this_total_p = psnr(real_cpu_np[j].transpose(1, 2, 0), recon_image_np[j].transpose(1, 2, 0))
        with open(PSNR_PATCHES, "a") as myfile:
            myfile.write('\n\t[Image %d] PSNR per Patch: %.4f | PSNR per Image: %.4f'
                         % (j + opt.batchSize * i, this_p, this_total_p))
        # print('[Image %d] PSNR per Patch: %.4f | PSNR per Image: %.4f' % (j + opt.batchSize*i, this_p, this_total_p))
        save_single_image(real_cpu[j], OUT_PATCHES, names_patches[j + opt.batchSize * i] + "_" + "real")
        save_single_image(recon_image.data[j], OUT_PATCHES, names_patches[j + opt.batchSize * i] + "_" + "recon")
        
        tot_psnr_patch_patches.append(this_p)
        tot_psnr_image_patches.append(this_total_p)





with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nHEALTHY MEAN PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (sum(tot_psnr_patch_healthy)/len(tot_psnr_patch_healthy), sum(tot_psnr_image_healthy)/len(tot_psnr_image_healthy)))

with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nUNHEALTHY MEAN PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (sum(tot_psnr_patch_unhealthy) / len(tot_psnr_patch_unhealthy),
                    sum(tot_psnr_image_unhealthy) / len(tot_psnr_image_unhealthy)))

with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nPATCHES MEAN PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (sum(tot_psnr_patch_patches) / len(tot_psnr_patch_patches),
                    sum(tot_psnr_image_patches) / len(tot_psnr_image_patches)))

with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nTOTAL MEAN PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % ((sum(tot_psnr_patch_healthy) + sum(tot_psnr_patch_unhealthy)) / (len(tot_psnr_patch_healthy) + len(tot_psnr_patch_unhealthy)),
                    (sum(tot_psnr_image_healthy) + sum(tot_psnr_image_unhealthy)) / (len(tot_psnr_image_healthy) + len(tot_psnr_image_unhealthy))))



with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\n\nHEALTHY STD PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (np.std(np.array(tot_psnr_patch_healthy)), np.std(np.array(tot_psnr_image_healthy))))

with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nUNHEALTHY STD PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (np.std(np.array(tot_psnr_patch_unhealthy)),
                    np.std(np.array(tot_psnr_image_unhealthy))))

with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nPATCHES STD PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (np.std(np.array(tot_psnr_patch_patches)),
                    np.std(np.array(tot_psnr_image_patches))))

with open(PSNR_TOTAL, "a") as myfile:
    myfile.write('\nTOTAL STD PSNRs: PSNR per Patch: %.4f | PSNR per Image: %.4f'
                 % (np.std(np.array(tot_psnr_patch_unhealthy + tot_psnr_patch_healthy)),
                    np.std(np.array(tot_psnr_image_unhealthy + tot_psnr_image_healthy))))
    

print("Done, see results in ", opt.output)

