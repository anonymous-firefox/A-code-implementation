from cgitb import enable

import torch
import torch.nn as nn
from torch.nn import ModuleList
from model.convlstm2 import ChannelAdjuster

import torch
# import gc
# import inspect
# import sys


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWConv, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                                 groups=in_channels)

    def forward(self, x):
        return self.dw_conv(x)


class DWDConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=7,dilation=3):
        super(DWDConv, self).__init__()
        padding=int(dilation*(kernel_size-1)/2)
        self.dw_d_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, dilation=dilation)

    def forward(self, x):
        return self.dw_d_conv(x)


class TimeAttention(nn.Module):
    def __init__(self, in_channels):
        super(TimeAttention, self).__init__()

        self.SA= nn.Sequential(
            DWConv(in_channels, in_channels),
            DWDConv(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.DA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        sa = self.SA(x)
        da = self.DA(x)
        return self.sigmoid(sa * da)* x


class TAU3(nn.Module):
    def __init__(self, in_channels, out_channels=1,input_length=12,output_length=4,hidden_dim=128):#512?
        super(TAU3, self).__init__()
        self.dim=hidden_dim
        self.mode='onetap'
        # self.mode = 'iterate'

        self.input_length=input_length
        self.output_length=output_length

        self.adjust_atmosphere = ChannelAdjuster(in_channels -1, self.dim -  out_channels)
        self.adjust_input= ChannelAdjuster(self.dim, self.dim)
        self.adjust_input2 = ChannelAdjuster(in_channels, self.dim)


        self.encoder1 =nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        self.encoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1),
        )


        self.time_attentions = ModuleList([TimeAttention(self.dim * self.input_length) for _ in range(4)])


        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.dim, self.dim, kernel_size=3,  padding=1),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(self.dim, self.dim, kernel_size=3,  padding=1),
        )
        self.decoder2 = nn.ConvTranspose2d(self.dim, self.dim, kernel_size=3, padding=1)

        self.get_next_conv = nn.Conv3d(self.input_length, 1, kernel_size=1, stride=1, padding=0)
        self.adjust_output = ChannelAdjuster(self.dim,  out_channels)
        self.adjust_output_onetap = ChannelAdjuster(self.dim*self.input_length//self.output_length, out_channels)

    def forward(self, atmosphere, pollution):

        if pollution is not None:
            atmosphere=atmosphere[:,:self.input_length]
            pollution=pollution[:,:self.input_length]
            x  = torch.cat([self.adjust_atmosphere(atmosphere), pollution], dim=2)
            x = self.adjust_input(x).contiguous()
        else:
            x=atmosphere
            x= self.adjust_input2(x).contiguous()


        B, T, C, H, W = x.shape
        if self.mode == 'iterate':
            results=x

            for i in range(self.output_length):
                torch.cuda.empty_cache()
                x=results[:,-self.input_length:].contiguous()
                x = x.view(B * T, C, H, W)
                x_res = self.encoder1(x)
                x=self.encoder2(x_res)

                x = x.view(B , T*C, H, W)
                for attention in self.time_attentions:
                    x = attention(x)+x

                x = x.view(B * T, C, H, W)
                x = self.decoder1(x)
                x=self.decoder1(x+x_res)

                result=x.view(B, T, C, H, W)
                result=self.get_next_conv(result)
                results = torch.cat(( results, result), dim=1)

            results=results[:,-self.output_length:]
            results = self.adjust_output(results)
            results=results.squeeze(2)

        elif self.mode=='onetap':
            x = x.view(B * T, C, H, W)
            x_res = self.encoder1(x)
            x = self.encoder2(x_res)

            x = x.view(B, T * C, H, W)
            for attention in self.time_attentions:
                x = attention(x) + x

            x = x.view(B * T, C, H, W)
            x = self.decoder1(x)
            x = self.decoder2(x)

            results = x.view(B*self.output_length, -1, H, W)
            results = self.adjust_output_onetap(results)
            if pollution is None:
                return results
            # print(results.shape)
            results = results.view(B, self.output_length, -1, H, W).squeeze(2)
            # print(results.shape)
        return results



