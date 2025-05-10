import torch.nn as nn
import torch

from model.convlstm2 import ConvLSTM,ChannelAdjuster
from model.ODE import compute_advection, compute_laplacian
from model.TAU3 import TAU3
class ODETAU(nn.Module):
    def __init__(self, input_length, output_length,atmosphere_size,hidden_dim=32,if_difussion=1,if_advection=1,if_product=1):
        super(ODETAU, self).__init__()
        self.c,self.h,self.w=atmosphere_size

        self.k_dif = nn.Parameter(torch.tensor(50, dtype=torch.float))
        self.dx = 50  # km
        self.dy = 55  # km
        self.dt = 6  # h

        self.if_difussion =if_difussion
        self.if_advection=if_advection
        self.if_product = if_product

        self.input_length = input_length
        self.output_length = output_length

        self.hidden_dim=hidden_dim
        self.odedim=self.if_difussion + self.if_advection + self.if_product + 1
        self.adjust_atmosphere = ChannelAdjuster(self.c,self.hidden_dim-1)
        self.adjust_result=ChannelAdjuster(self.hidden_dim+self.odedim,1)
        self.adjust_odeandatmos = ChannelAdjuster(self.hidden_dim-1+self.odedim, 1)


        self.windConvLSTM = ConvLSTM(input_dim=2, hidden_dim=[8,2], structure='linear') #TAU is also OK

        self.TAU=TAU3(in_channels=self.hidden_dim+self.odedim,
                      out_channels=self.hidden_dim-1,
                      input_length=self.input_length,
                      output_length=1,
                      hidden_dim=self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def calculate(self,pollut, tensor_now, v_x, v_y,if_difussion=1,if_advection=1,if_product=1):
        features = []
        if if_difussion:
            laplacian_X = compute_laplacian(pollut, self.dx, self.dy)
            difussion = (self.dt * self.k_dif * laplacian_X).unsqueeze(1)
            features.append(difussion)
        if if_advection:
            advection = compute_advection(pollut, v_x, v_y, self.dx, self.dy)
            advection = self.dt * (-advection)
            features.append(advection.unsqueeze(1))
        if 1:
            atmosphere_feature = self.adjust_result(tensor_now).squeeze(1)
            features.append(atmosphere_feature.unsqueeze(1))
        if if_product:
            product = self.sigmoid(atmosphere_feature) * pollut
            features.append(product.unsqueeze(1))
        return torch.cat(features, dim=1).unsqueeze(1)

    def forward(self, atmosphere, pollution):
        wind_current_input = atmosphere[:, :self.input_length, 265:267]
        wind_hidden_state = None
        wind_total = wind_current_input


        adjusted_atmosphere=self.adjust_atmosphere(atmosphere)
        tau_result  = torch.cat([adjusted_atmosphere, pollution], dim=2)[:,:self.input_length]
        total=tau_result

        result_list = [torch.zeros((total.shape[0], 1, self.odedim, self.h,self.w), device=atmosphere.device)]
        for i in range(self.input_length-1):
            v_x = atmosphere[:, :self.input_length, 265]
            v_y = -atmosphere[:, :self.input_length, 266]
            ODE=result_list[-1].squeeze(1)
            tensor=torch.cat((ODE, total[:, i]), dim=1)

            result = self.calculate(pollution[:, i].squeeze(1),
                                    tensor,
                                    v_x[:, i],
                                    v_y[:, i],
                                    if_difussion=self.if_difussion,
                                    if_advection=self.if_advection,
                                    if_product=self.if_product)

            result_list.append(result)

        result_concat = torch.cat(result_list, dim=1)
        total= torch.cat((result_concat,total), dim=2)#8,12,32+4,80,130

        for i in range(self.output_length):
            pollut = total[:, -1,-1]

            wind_output, wind_hidden_state = self.windConvLSTM(wind_total[:, -self.input_length:], wind_hidden_state)
            wind_current_input = wind_output[-1][:, -1:]
            wind_total = torch.cat((wind_total, wind_current_input), dim=1)
            wind_now=wind_total[:, self.input_length - 1 + i]
            v_x ,v_y = wind_now[:, 0] ,- wind_now[:, 1] # 8, 80, 130
            # V is by default oriented from south to north, which does not align with our coordinate system

            result = self.calculate(pollut,
                                    total[:,self.input_length+i-1],
                                    v_x,
                                    v_y,
                                    if_difussion=self.if_difussion,
                                    if_advection=self.if_advection,
                                    if_product=self.if_product)

            tau_result = self.TAU(total[:,-self.input_length:],None).unsqueeze(1)

            concat=torch.cat((result,tau_result), dim=2)
            pollution_result= self.adjust_odeandatmos(concat)
            result_now=torch.cat((concat,pollution_result), dim=2)

            total = torch.cat((total, result_now), dim=1)

        result=total[:,-self.output_length:,-1]
        return result