import torch
import numpy as np
import torch.nn as nn

######### SIMVP ########

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y

########## UNET ############
class CONVB(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.pass1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(output_channels),
                                  nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(output_channels),
                                  nn.ReLU())
    def forward(self, inputs):
        return self.pass1(inputs)

class UP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.pass1 = CONVB(input_channels, output_channels)
        self.pass2 = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.pass1(inputs)
        y = self.pass2(x)
        return (x,y)

class DOWN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.pass1 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, padding=0)
        self.pass2 = CONVB(output_channels+output_channels, output_channels)


    def forward(self, inputs, aux):
        x = self.pass1(inputs)
        x = torch.cat([x, aux], axis=1)
        x = self.pass2(x)

        return x

class OUTC(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.pass1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        return self.pass1(inputs)

class UPSTREAM(nn.Module):
    def __init__(self, input_channels,mc1,mc2,mc3,con1,output_channels):
        super().__init__()

        self.pass1 = UP(input_channels,mc1)
        self.pass2 = UP(mc1,mc2)
        self.pass3 = UP(mc2,mc3)
        self.pass4 = UP(mc3,con1)
        self.pass5 = CONVB(con1,output_channels)

    def forward(self, inputs):
        x1,y1 = self.pass1(inputs)
        x2,y2 = self.pass2(y1)
        x3,y3 = self.pass3(y2)
        x4,y4 = self.pass4(y3)
        z = self.pass5(y4)
        return z,x1,x2,x3,x4

class DOWNSTREAM(nn.Module):
    def __init__(self, input_channels,mc1,mc2,mc3,output_channels):
        super().__init__()
        self.d1 = DOWN(input_channels, mc1)
        self.d2 = DOWN(mc1, mc2)
        self.d3 = DOWN(mc2, mc3)
        self.d4 = DOWN(mc3, output_channels)

    def forward(self,a,b,c,d,e):
        x1 = self.d1(a,e)
        x2 = self.d2(x1,d)
        x3 = self.d3(x2,c)
        x4 = self.d4(x3,b)
        return x4


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pass1 = UPSTREAM(3,64,128,256,512,1024)
        self.pass2 = DOWNSTREAM(1024,512,256,128,64)
        self.pass3 = OUTC(64,49)

    def forward(self, inputs):
        x,y1,y2,y3,y4 = self.pass1(inputs)
        x = self.pass2(x,y1,y2,y3,y4)
        x = self.pass3(x)
        return x
		
### FINAL MODEL ###

class vdU_pred(nn.Module):
    def __init__(self):
        super(vdU_pred, self).__init__()
        self.vp = SimVP((11,3,160,240))
        self.un = UNet()

    def forward(self,x):
        pred_vid = self.vp(x)
        seg = self.un(pred_vid[:,10,:,:,:])
        return seg

    def pred_vid(self,x):
        return self.vp(x)
    def fix_vp(self):
        for param in self.vp.parameters():
            param.requires_grad = False

    def fix_un(self):
        for param in self.un.parameters():
            param.requires_grad = False

    def unfix_vp(self):
        for param in self.vp.parameters():
            param.requires_grad = True

    def unfix_un(self):
        for param in self.un.parameters():
            param.requires_grad = True

    def load_weights(self,fpathv,fpathu,device):
        PATH = fpathv
        self.vp.load_state_dict(torch.load(PATH,map_location=device))
        PATH = fpathu
        self.un.load_state_dict(torch.load(PATH,map_location=device))