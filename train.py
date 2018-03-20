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

from model import _netlocalD, _netG
from plotter import plotter
from psnr import psnr

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--test_workers', type=int, help='number of data loading workers on testset', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the patch to be reconstructed')
parser.add_argument('--beforeCropSize', type=int, default=1024,
                    help='the height / width of the rescaled image before eventual cropping')
parser.add_argument('--name', default='default_experiment', help='the name of the experiment used for the directory')

# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=1)  # By default convert image input and output to grayscale
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--disc_per_gen_factor', type=int, default=1, help='number of discriminator iterations every generator iteration')
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

opt = parser.parse_args()
opt.cuda = True

# opt.ndf = 128 #Discriminator
# opt.nef = 128 #Generator
# opt.imageSize = 128
# opt.patchSize = 64
# opt.beforeCropSize = 128
opt.randomCrop = True
opt.continueTraining = False

STEPS_TO_PLOT = 200  # how often to update plots

# Path parameters
if opt.randomCrop:
    opt.name = "randomCrop"
EXP_NAME = "".join(
    [opt.name, '_imageSize', str(opt.imageSize), '_patchSize', str(opt.patchSize), '_nef', str(opt.nef), '_ndf',
     str(opt.ndf)])
PATH_netG = "outputs/" + EXP_NAME + "/netG_context_encoder.pth"
PATH_netD = "outputs/" + EXP_NAME + "/netD_discriminator.pth"
PATH_measures = "outputs/" + EXP_NAME + "/measures.pickle"
PATH_train = "outputs/" + EXP_NAME + "/train_results"
PATH_test = "outputs/" + EXP_NAME + "/test_results"
PATH_plots = "outputs/" + EXP_NAME + "/plots"
PATH_randomCrops = "outputs/" + EXP_NAME + "/test_results/randomCrops"

if opt.continueTraining:
    print("Continuing with the training of the existing model in:", "./outputs/" + EXP_NAME)

print(opt)

print("\nStarting Experiment:", EXP_NAME, "\n")

try:
    os.makedirs("outputs")
except OSError:
    pass
try:
    os.makedirs("outputs/" + EXP_NAME)
except OSError:
    pass
try:
    os.makedirs(PATH_plots)
except OSError:
    pass
try:
    os.makedirs(PATH_train)
except OSError:
    pass
try:
    os.makedirs(PATH_test)
except OSError:
    pass
if opt.randomCrop:
    try:
        os.makedirs(PATH_randomCrops)
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
        transforms.Resize(opt.beforeCropSize),
        transforms.CenterCrop(opt.CENTER_SIZE_randomCrop),
        transforms.RandomCrop(opt.imageSize, opt.PAD_randomCrop),
        transforms.ToTensor(),
    ])
    transform_original = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(opt.beforeCropSize),
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
        transforms.Resize(opt.beforeCropSize),
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


def save_image(real_cpu, recon_image, epoch, PATH, addition=""):
    vutils.save_image(real_cpu,
                      PATH + '/epoch_%03d_' % epoch + addition + 'real.png')
    vutils.save_image(recon_image,
                      PATH + '/epoch_%03d_' % epoch + addition + 'recon.png')


resume_epoch = 0

netG = _netG(opt)
netG.apply(weights_init)
if opt.continueTraining:
    print("Loading model netG from: ", PATH_netG)
    netG.load_state_dict(torch.load(PATH_netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(PATH_netG)['epoch']
print(netG)

netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.continueTraining:
    print("Loading model netD from: ", PATH_netD)
    netD.load_state_dict(torch.load(PATH_netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(PATH_netD)['epoch']
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

real_center = torch.FloatTensor(opt.batchSize, 1, int(opt.imageSize / 2), int(opt.imageSize / 2))

print("Moving models to CUDA...")

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

real_center = Variable(real_center)

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
    (D_G_zs, D_xs, Advs, L2s, G_tots, D_tots) = pickle.load(open(PATH_measures, "rb"))
    print("Loaded saved measures with ", len(D_G_zs), "datapoints, approximately ",
          math.ceil(len(D_G_zs) * STEPS_TO_PLOT / len(dataloader)), "epochs")

this_DGz = 0
this_Dx = 0
this_Adv = 0
this_L2 = 0
this_G_tot = 0
this_D_tot = 0

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
        
        # train with real
        netD.zero_grad()
        label.data.resize_(batch_size, 1).fill_(real_label)
        
        # input("Proceed..." + str(real_center.data.size()))
        
        # print(real_center.data.size())
        output = netD(real_center)
        # print(output.data.size())
        # print(label.data.size())
        # input()
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()
        
        # train with fake

        # print(input_cropped.size())
        fake = netG(input_cropped)
        # print(fake.data.size(), " ", input_cropped.data.size())
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        # print(output.data.size(), " ", label.data.size())
        # input("")
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if i % opt.disc_per_gen_factor == 0:  # Step Generator every n Discriminator steps
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
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
        
        if step_counter == STEPS_TO_PLOT:
            this_Adv *= (1 - wtl2)
            this_L2 *= wtl2
            this_DGz /= STEPS_TO_PLOT
            this_Dx /= STEPS_TO_PLOT
            this_Adv /= STEPS_TO_PLOT
            this_L2 /= STEPS_TO_PLOT
            this_G_tot /= STEPS_TO_PLOT
            this_D_tot /= STEPS_TO_PLOT
            
            D_G_zs.append(this_DGz)
            D_xs.append(this_Dx)
            Advs.append(this_Adv)
            L2s.append(this_L2)
            G_tots.append(this_G_tot)
            D_tots.append(this_D_tot)
            
            plotter(D_G_zs, D_xs, Advs, L2s, G_tots, D_tots, (len(dataloader) / STEPS_TO_PLOT), PATH_plots)
            
            this_DGz = 0
            this_Dx = 0
            this_Adv = 0
            this_L2 = 0
            this_G_tot = 0
            this_D_tot = 0
            step_counter = 0
        
        if i % 100 == 0:
            recon_image = input_cropped.clone()
            recon_image.data[:, :,
            int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2),
            int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)] = fake.data
            save_image(real_cpu, recon_image.data, epoch, PATH_train)
    
    #####################################
    # Testing at the end of every epoch #
    #####################################
    
    if epoch == 0:
        # Clear existing file if any
        with open(PATH_test + "/PSNRs.txt", "w") as myfile:
            myfile.write("")
            
    with open(PATH_test + "/PSNRs.txt", "a") as myfile:
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

        with open(PATH_test + "/PSNRs.txt", "a") as myfile:
            myfile.write('\n\t[%d/%d] PSNR per Patch: %.4f | PSNR per Image: %.4f'
              % (i + 1, len(test_dataloader), p / opt.batchSize, total_p / opt.batchSize))
        
        save_image(real_cpu, recon_image.data, epoch, PATH_test)
        
    if opt.randomCrop:
        MIN = (opt.beforeCropSize - opt.CENTER_SIZE_randomCrop) // 2
        MAX = (opt.beforeCropSize + opt.CENTER_SIZE_randomCrop) // 2 - opt.imageSize
        input_rc_cropped = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
        input_rc_cropped = Variable(input_rc_cropped.cuda())
        for data in test_original_dataloader:
            for k in range(2):
                image_1024 = data[0][k].view(-1, 1, opt.beforeCropSize, opt.beforeCropSize)
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
                    int(opt.imageSize / 2 - opt.patchSize / 2):int(opt.imageSize / 2 + opt.patchSize / 2)] = fake.data[0].view(-1, 1, opt.patchSize, opt.patchSize)
                    save_image(real_cpu, recon_image.data, epoch, PATH_randomCrops, addition=str(k) + "_sub_"+str(l))
                    save_image(input_rc_cropped.expand(opt.batchSize, -1, -1, -1).data, fake.data, epoch, PATH_randomCrops, addition=str(k) + "_batches_" + str(l))
                    
                    image_1024_recon[:, :, int(x + opt.imageSize/2 - opt.patchSize/2):int(x + opt.imageSize/2 + opt.patchSize/2), int(y + opt.imageSize/2 - opt.patchSize/2):int(y + opt.imageSize/2 + opt.patchSize/2)] = fake.data[0].view(-1, 1, opt.patchSize, opt.patchSize)
                save_image(image_1024, image_1024_recon, epoch, PATH_randomCrops, addition=str(k) + "_")

    # Store model checkpoint
    torch.save({'epoch': epoch + 1, 'state_dict': netG.state_dict()}, PATH_netG)
    torch.save({'epoch': epoch + 1, 'state_dict': netD.state_dict()}, PATH_netD)

    # Store measure lists
    pickle.dump((D_G_zs, D_xs, Advs, L2s, G_tots, D_tots), open(PATH_measures, "wb"))
    
    print("\tEpoch", epoch + 1, "took ", (time.time() - epoch_time) / 60, "minutes")
