import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.multiplierG = opt.imageSize / (opt.patchSize * 2)
        self.ngpu = opt.ngpu
        
        main = nn.Sequential()
        
        # Conv
        
        main.add_module(
            'ENC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(opt.imageSize, opt.imageSize // 2, opt.nc, opt.ndf),
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False))
        main.add_module('ENC_imsize.{0}_depth.{1}.lrelu'.format(opt.imageSize // 2, opt.ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = int(opt.imageSize / 2), opt.ndf
        
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('ENC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, csize // 2, in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('ENC_imsize.{0}_depth.{1}.batchnorm'.format(csize // 2, out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('ENC_imsize.{0}_depth.{1}.lrelu'.format(csize // 2, out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize // 2
        
        bottleneck = cndf * 2
        csize = int(csize)
        
        main.add_module('ENC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, 1, cndf, bottleneck),
                        nn.Conv2d(cndf, bottleneck, 4, 1, 0, bias=False))
        main.add_module('ENC_imsize.{0}_depth.{1}.batchnorm'.format(1, bottleneck),
                        nn.BatchNorm2d(bottleneck))
        main.add_module('ENC_imsize.{0}_depth.{1}.lrelu'.format(1, bottleneck),
                        nn.LeakyReLU(0.2, inplace=True))

        # Deconv
        
        ngf = opt.imageSize / opt.patchSize * opt.nef / 2
        cngf, tisize = ngf // 2, 4
        while tisize != opt.imageSize:
            cngf = cngf * 2
            tisize = tisize * 2
        
        cngf = int(cngf)
        tisize = int(tisize)
        csize = 4
        
        main.add_module('DEC_imsize.{0}-{1}_depth.{2}-{3}.deconv2d'.format(1, csize, bottleneck, cngf),
                        nn.ConvTranspose2d(bottleneck, cngf, 4, 1, 0, bias=False))
        main.add_module('DEC_imsize.{0}_depth.{1}.batchnorm'.format(csize, cngf), nn.BatchNorm2d(cngf))
        main.add_module('DEC_imsize.{0}_depth.{1}.relu'.format(csize, cngf), nn.ReLU(True))
        
        while csize < opt.patchSize // 2:
            main.add_module('DEC_imsize.{0}-{1}_depth.{2}-{3}.deconv2d'.format(csize, csize * 2, cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('DEC_imsize.{0}_depth.{1}.batchnorm'.format(csize * 2, cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('DEC_imsize.{0}_depth.{1}.relu'.format(csize * 2, cngf // 2),
                            nn.ReLU(True))
            
            cngf = cngf // 2
            csize = csize * 2
        
        main.add_module('DEC_imsize.{0}-{1}_depth.{2}-{3}.deconv2d'.format(csize, csize * 2, cngf, opt.nc),
                        nn.ConvTranspose2d(cngf, opt.nc, 4, 2, 1, bias=False))
        main.add_module('DEC_imsize.{0}_depth.{1}.final_tanh'.format(csize * 2, opt.nc),
                        nn.Tanh())
        
        self.main = main
        
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
        
        main = nn.Sequential()
        
        main.add_module(
            'DISC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(opt.patchSize, opt.patchSize // 2, opt.nc, opt.nef),
            nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False))
        main.add_module('DISC_imsize.{0}_depth.{1}.lrelu'.format(opt.patchSize // 2, opt.nef),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cnef = int(opt.patchSize / 2), opt.nef
        
        while csize > 4:
            in_feat = cnef
            out_feat = cnef * 2
            main.add_module('DISC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, csize // 2, in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('DISC_imsize.{0}_depth.{1}.batchnorm'.format(csize // 2, out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('DISC_imsize.{0}_depth.{1}.lrelu'.format(csize // 2, out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cnef = cnef * 2
            csize = csize // 2
        
        csize = int(csize)
        
        main.add_module('DISC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, 1, cnef, 1),
                        nn.Conv2d(cnef, 1, csize, 1, 0, bias=False))
        main.add_module('DISC_imsize.{0}_depth.{1}.final_sigmoid'.format(1, 1),
                        nn.Sigmoid())
        
        self.main = main
        
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        
        return output.view(-1, 1)



class _netmarginD(nn.Module):
    def __init__(self, opt):
        super(_netmarginD, self).__init__()
        self.ngpu = opt.ngpu
        
        main = nn.Sequential()
        
        main.add_module(
            'DISC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(opt.patch_with_margin_size, opt.patch_with_margin_size // 2, opt.nc, opt.nef),
            nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False))
        main.add_module('DISC_imsize.{0}_depth.{1}.lrelu'.format(opt.patch_with_margin_size // 2, opt.nef),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cnef = int(opt.patch_with_margin_size / 2), opt.nef
        
        while csize > 4:
            in_feat = cnef
            out_feat = cnef * 2
            main.add_module('DISC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, csize // 2, in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('DISC_imsize.{0}_depth.{1}.batchnorm'.format(csize // 2, out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('DISC_imsize.{0}_depth.{1}.lrelu'.format(csize // 2, out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cnef = cnef * 2
            csize = csize // 2
        
        csize = int(csize)
        # print("csize", csize)
        
        main.add_module('DISC_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, 1, cnef, 1),
                        nn.Conv2d(cnef, 1, csize, 1, 0, bias=False))
        main.add_module('DISC_imsize.{0}_depth.{1}.final_sigmoid'.format(1, 1),
                        nn.Sigmoid())
        
        self.main = main
    
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            
        return output.view(-1, 1)



class _netjointD(nn.Module):
    def __init__(self, opt):
        super(_netjointD, self).__init__()
        self.ngpu = opt.ngpu
        
        
        # Local Disc
        
        main_local = nn.Sequential()
        
        main_local.add_module(
            'DISClocal_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(opt.patch_with_margin_size, opt.patch_with_margin_size // 2, opt.nc, opt.nef),
            nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=False))
        main_local.add_module('DISClocal_imsize.{0}_depth.{1}.lrelu'.format(opt.patch_with_margin_size // 2, opt.nef),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cnef = int(opt.patch_with_margin_size / 2), opt.nef
        
        while csize > 4:
            in_feat = cnef
            out_feat = cnef * 2
            main_local.add_module('DISClocal_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, csize // 2, in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main_local.add_module('DISClocal_imsize.{0}_depth.{1}.batchnorm'.format(csize // 2, out_feat),
                            nn.BatchNorm2d(out_feat))
            main_local.add_module('DISClocal_imsize.{0}_depth.{1}.lrelu'.format(csize // 2, out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cnef = cnef * 2
            csize = csize // 2
        
        csize = int(csize)
        
        main_local.add_module('DISClocal_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, 1, cnef, opt.fullyconn_size),
                        nn.Conv2d(cnef, opt.fullyconn_size, csize, 1, 0, bias=False))
        
        self.main_local = main_local


        # Global Disc

        main_global = nn.Sequential()

        main_global.add_module(
            'DISCglobal_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(opt.imageSize, opt.imageSize // 4, opt.nc, opt.nef),
            nn.Conv2d(opt.nc, opt.nef, 8, 4, 2, bias=False))
        main_global.add_module('DISCglobal_imsize.{0}_depth.{1}.lrelu'.format(opt.imageSize // 4, opt.nef),
                              nn.LeakyReLU(0.2, inplace=True))
        csize, cnef = int(opt.imageSize / 4), opt.nef

        while csize > 4:
            in_feat = cnef
            out_feat = cnef * 2
            main_global.add_module(
                'DISCglobal_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, csize // 4, in_feat, out_feat),
                nn.Conv2d(in_feat, out_feat, 8, 4, 2, bias=False))
            main_global.add_module('DISCglobal_imsize.{0}_depth.{1}.batchnorm'.format(csize // 4, out_feat),
                                  nn.BatchNorm2d(out_feat))
            main_global.add_module('DISCglobal_imsize.{0}_depth.{1}.lrelu'.format(csize // 4, out_feat),
                                  nn.LeakyReLU(0.2, inplace=True))
            cnef = cnef * 2
            csize = csize // 4

        csize = int(csize)

        main_global.add_module('DISCglobal_imsize.{0}-{1}_depth.{2}-{3}.conv2d'.format(csize, 1, cnef, opt.fullyconn_size),
                              nn.Conv2d(cnef, opt.fullyconn_size, csize, 1, 0, bias=False))

        self.main_global = main_global
        
        
        # Joint Discriminator

        main_joint = nn.Sequential()
        
        main_joint.add_module('DISCjoint_imsize.{0}-{1}.fully_connected'.format(opt.fullyconn_size*2, 1),
            nn.Linear(opt.fullyconn_size * 2, 1, bias=False))
        main_joint.add_module('DISCjoint_imsize.{0}_depth.{1}.final_sigmoid'.format(1, 1),
                              nn.Sigmoid())
        
        self.main_joint = main_joint
        
    
    def forward(self, input_center, input_real):
        if isinstance(input_center.data, torch.cuda.FloatTensor) and isinstance(input_real.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            print("Joint discriminator on multiple GPU wasn't tested.")
            output_local = nn.parallel.data_parallel(self.main_local, input_center, range(self.ngpu))
            output_global = nn.parallel.data_parallel(self.main_global, input_real, range(self.ngpu))
            output_joint = torch.cat((output_global, output_local), dim=1).view(output_local.size(0),
                                                                                output_local.size(1) * 2)
            
            output = nn.parallel.data_parallel(self.main_joint, output_joint, range(self.ngpu))
        else:
            output_local = self.main_local(input_center)
            output_global = self.main_global(input_real)
            output_joint = torch.cat((output_global, output_local), dim=1).view(output_local.size(0), output_local.size(1)*2)
            
            # print(output_global.size(), type(output_global))
            # print(output_local.size(), type(output_local))
            # print(output_joint.size(), type(output_joint))
            # input()
            output = self.main_joint(output_joint)
        
        return output.view(-1, 1)
