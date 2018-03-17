import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.multiplierG = opt.imageSize / (opt.patchSize * 2)
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc, opt.nef, int(4 * self.multiplierG), int(2 * self.multiplierG), int(1 * self.multiplierG), bias=False), #modified
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 64 x 64
            nn.Conv2d(opt.nef, opt.nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 32 x 32
            nn.Conv2d(opt.nef, opt.nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(opt.nef * 2, opt.nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 8 x 8
            nn.Conv2d(opt.nef * 4, opt.nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*8) x 4 x 4
            nn.Conv2d(opt.nef * 8, opt.nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*16) x 2 x 2
            nn.Conv2d(opt.nef * 16, opt.nef * 32, 4, 2, 1, bias=False),
            # state size: (nef*32) x 1 x 1
            nn.BatchNorm2d(opt.nef * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # input is nef * 32, going into a convolution
            nn.ConvTranspose2d(opt.nef * 32, opt.nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2
            nn.ConvTranspose2d(opt.nef * 16, opt.nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.nef * 8, opt.nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.nef * 4, opt.nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.nef * 2, opt.nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.nef, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netlocalD(nn.Module):
    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        self.multiplierD = opt.patchSize / 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, int(4 * self.multiplierD), int(2 * self.multiplierD), int(1 * self.multiplierD), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        # print(output.data)
        
        return output.view(-1, 1)
