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
import matplotlib.pyplot as plt
import math

from model import _netlocalD, _netG
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lungs', help='streetview | tiny-imagenet | lungs ')
parser.add_argument('--dataroot', default='', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
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
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
parser.add_argument('--wtl2', type=float, default=0.998, help='0 means do not use else use with this weight')
parser.add_argument('--wtlD', type=float, default=0.001, help='0 means do not use else use with this weight')

opt = parser.parse_args()
opt.cuda = True

opt.wtl2 = 0
opt.ndf = 128
opt.nef = 128
CONTINUE_TRAINING = False

if CONTINUE_TRAINING:
    print("Continuing with the training of an existing model")
    opt.netD = "model/netlocalD.pth"
    opt.netG = "model/netG_streetview.pth"

print(opt)

try:
    os.makedirs("plots")
    os.makedirs("model")
    os.makedirs('result/' + str(opt.dataset) + '/cropped')
    os.makedirs('result/' + str(opt.dataset) + '/real')
    os.makedirs('result/' + str(opt.dataset) + '/recon')
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = 1234
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'tiny-imagenet':
    # folder dataset
    dataset = dset.ImageFolder(root='dataset_tiny_imagenet/train',
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lungs':
    # folder dataset
    if opt.nc == 1:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = dset.ImageFolder(root='dataset_lungs/train', transform=transform)
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    dataset = dset.ImageFolder(root="dataset/train", transform=transform)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10


# plot losses on a unique figure 'plot.png'
def plotter(D_G_zs, D_xs, Advs, L2s, G_tots, D_tots):
    x = list(range(len(D_tots)))
    log_4 = [-math.log(4)] * len(D_tots)
    D_gain = [-k for k in D_tots]  # Discriminator gain defined as negative cross-entropy
    
    vline_position = [len(dataloader)/STEPS_TO_PLOT * (x + 1) for x in range(int(math.floor(len(D_tots) * STEPS_TO_PLOT / len(dataloader))))]
    plt.clf()
    plt.plot(x, D_G_zs, "g-", linewidth=0.5, label="p D(G(z))")
    plt.plot(x, D_xs, "r-", linewidth=0.5, label="p D(x)")
    plt.plot(x, D_gain, "b-", linewidth=0.5, label="Disciminator")
    plt.plot(x, log_4, "k--", linewidth=0.5, label="-log(4)")
    plt.xlabel('x200 iterations')
    plt.ylabel('value')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/main4.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.clf()
    plt.plot(x, D_G_zs, "g-", linewidth=0.5, label="p D(G(z))")
    plt.plot(x, D_xs, "r-", linewidth=0.5, label="p D(x)")
    plt.xlabel('x200 iterations')
    plt.ylabel('loss')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/fake-real_probs.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.clf()
    plt.plot(x, Advs, "b-", linewidth=0.5, label="Adversarial loss")
    plt.plot(x, L2s, "g-", linewidth=0.5, label="L2 loss")
    plt.plot(x, G_tots, "k-", linewidth=0.5, label="Tot Generator loss")
    plt.xlabel('x200 iterations')
    plt.ylabel('loss')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/gen_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    plt.clf()
    plt.plot(x, D_tots, "b-", linewidth=0.5, label="Tot Discriminator loss")
    plt.xlabel('x200 iterations')
    plt.ylabel('loss')
    for k in vline_position:
        plt.axvline(x=k, linewidth=0.2, color='k', linestyle='--')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("plots/disc_losses.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    return


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch = 0

netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    print("Loading model netG from: ", opt.netG)
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)

netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    print("Loading model netD from: ", opt.netD)
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(opt.batchSize, 3, int(opt.imageSize / 2), int(opt.imageSize / 2))

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

# store information about losses for plotting
STEPS_TO_PLOT = 200 # how often to update plots
N_UPDATE_GEN = 1 # update generator every N discriminator updates
step_counter = 0

D_G_zs = []
D_xs = []
Advs = []
L2s = []
D_tots = []
G_tots = []

# Load measures from initial part of the training, if loading an existing model
if CONTINUE_TRAINING:
    (D_G_zs, D_xs, Advs, L2s, G_tots, D_tots) = pickle.load(open("measures.pickle", "rb"))
    print("Loaded saved measures with ", len(D_G_zs), "datapoints, approximately ",
          math.ceil(len(D_G_zs) / len(dataloader)), "epochs")

this_DGz = 0
this_Dx = 0
this_Adv = 0
this_L2 = 0
this_G_tot = 0
this_D_tot = 0

for epoch in range(resume_epoch, opt.niter):
    
    for i, data in enumerate(dataloader, 0):
        step_counter += 1
        
        real_cpu, _ = data
        real_center_cpu = real_cpu[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2),
                          int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2)]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped.data[:, 0,
        int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        int(opt.imageSize / 4 + opt.overlapPred):int(
            opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
        if opt.nc > 1:
            input_cropped.data[:, 1,
            int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
            int(opt.imageSize / 4 + opt.overlapPred):int(
                opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
            input_cropped.data[:, 2,
            int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
            int(opt.imageSize / 4 + opt.overlapPred):int(
                opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0
        
        # train with real
        netD.zero_grad()
        label.data.resize_(batch_size).fill_(real_label)
        
        # input("Proceed..." + str(real_center.data.size()))
        
        output = netD(real_center)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()
        
        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        fake = netG(input_cropped)
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
        if i % N_UPDATE_GEN == 0:  # Step Generator every 10 Discriminator steps
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG_D = criterion(output, label)
            # errG_D.backward(retain_variables=True)
            
            # errG_l2 = criterionMSE(fake,real_center)
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
              % (epoch, opt.niter, i+1, len(dataloader),
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
            
            plotter(D_G_zs, D_xs, Advs, L2s, G_tots, D_tots)

            this_DGz = 0
            this_Dx = 0
            this_Adv = 0
            this_L2 = 0
            this_G_tot = 0
            this_D_tot = 0
            step_counter = 0
        
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              'result/' + str(opt.dataset) + '/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                              'result/' + str(opt.dataset) + '/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_cropped.clone()
            recon_image.data[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2),
            int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2)] = fake.data
            vutils.save_image(recon_image.data,
                              'result/' + str(opt.dataset) + '/recon/recon_center_samples__epoch_%03d.png' % (epoch))
    
    # do checkpointing
    torch.save({'epoch': epoch + 1,
                'state_dict': netG.state_dict()},
               'model/netG_streetview.pth')
    torch.save({'epoch': epoch + 1,
                'state_dict': netD.state_dict()},
               'model/netlocalD.pth')
    
    # store measure lists
    t = (D_G_zs, D_xs, Advs, L2s, G_tots, D_tots)
    pickle.dump(t, open("measures.pickle", "wb"))
