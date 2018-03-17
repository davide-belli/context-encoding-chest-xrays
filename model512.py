import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 521 x 512
            nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 256 x 256
            nn.Conv2d(opt.nef, opt.nef*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 128 x 128
            nn.Conv2d(opt.nef*2, opt.nef*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 64 x 64
            nn.Conv2d(opt.nef*4, opt.nef*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*8) x 32 x 32
            nn.Conv2d(opt.nef*8, opt.nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*16) x 16 x 16
            nn.Conv2d(opt.nef * 16, opt.nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*32) x 8 x 8
            nn.Conv2d(opt.nef * 32, opt.nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*64) x 4 x 4
            nn.Conv2d(opt.nef * 64, opt.nef * 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*128) x 2 x 2
            nn.Conv2d(opt.nef * 128, opt.nef * 256, 4, 2, 1, bias=False),
            # state size: (nef*256 = 32k) x 1 x 1
            nn.BatchNorm2d(opt.nef * 256),
            nn.LeakyReLU(0.2, inplace=True),
            # input is nef * 256, going into a convolution
            nn.ConvTranspose2d(opt.nef * 256, opt.nef * 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 128),
            nn.ReLU(True),
            # state size. (nef*128) x 2 x 2
            nn.ConvTranspose2d(opt.nef * 128, opt.nef * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 64),
            nn.ReLU(True),
            # state size. (nef*64) x 4 x 4
            nn.ConvTranspose2d(opt.nef * 64, opt.nef * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 32),
            nn.ReLU(True),
            # state size. (nef*32) x 8 x 8
            nn.ConvTranspose2d(opt.nef * 32, opt.nef * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 16),
            nn.ReLU(True),
            # state size. (nef*16) x 16 x 16
            nn.ConvTranspose2d(opt.nef * 16, opt.nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 8),
            nn.ReLU(True),
            # state size. (nef*8) x 32 x 32
            nn.ConvTranspose2d(opt.nef * 8, opt.nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.nef * 4),
            nn.ReLU(True),
            # state size. (nef*4) x 64 x 64
            nn.ConvTranspose2d(opt.nef, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
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
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(opt.ndf * 16, opt.ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 2 x 2
            nn.Conv2d(opt.ndf * 32, 1, 4, 2, 1, bias=False),
            # state size. (1) x 1 x 1
            nn.Sigmoid()
        )
    
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        # print(output.data)
        
        return output.view(-1, 1)
