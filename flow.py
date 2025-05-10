from tqdm import tqdm
from utils import *
from cfg import cfg
import time


params = load_normalization_params('dataset/saved/normalization_params_.pt')
apply_normalization = Apply_Normalization(params)
import torch

def flow(model, loader, mode, optimizer=None):
    if mode == 'Training':
        model.train()
        loss = 0.0
        for atmosphere, pollution, groundtruth in tqdm(loader, desc=' '+mode, total=len(loader), position=0):
            if cfg.tik:
                print(f'Model loaded after {time.time() - cfg.tik:.1f}s')
                cfg.tik=0
            if atmosphere.shape[0]==1:
                continue
            atmosphere, pollution, groundtruth = apply_normalization(atmosphere, pollution, groundtruth)#8,4,354/1/1,80,130
            optimizer.zero_grad()
            outputs = model(atmosphere, pollution)

            loss =cfg.criterion(*calculate_criterion(outputs, pollution, groundtruth))
            # print(loss)
            loss.backward()

            if cfg.ouput_length>4:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

            optimizer.step()
        return loss, optimizer

    else:
        model.eval()
        squared_loss = 0.0
        absolute_loss = 0.0
        smape_loss=0
        total_number=0
        with (torch.no_grad()):
            for atmosphere, pollution, groundtruth in tqdm(loader, desc=' '+mode, total=len(loader), position=0):
                if cfg.tik:
                    print(f'Model loaded after {time.time() - cfg.tik:.1f}s')
                    cfg.tik = 0

                atmosphere, pollution, groundtruth = apply_normalization(atmosphere, pollution, groundtruth)
                outputs = model(atmosphere, pollution)
                # outputs = model(atmosphere, torch.nan_to_num(groundtruth, nan=0))
                if mode == 'Testing':
                    outputs = apply_normalization.reverse_normalize(outputs)
                    pollution = apply_normalization.reverse_normalize(pollution)
                    groundtruth = apply_normalization.reverse_normalize(groundtruth)
                # print(groundtruth.size)
                total_number+=groundtruth.size(0)
                result,ground=calculate_criterion(outputs, pollution, groundtruth)

                squared_loss += groundtruth.size(0) * cfg.criterion(result,ground).item()
                if mode == 'Testing':
                    absolute_loss += groundtruth.size(0) * torch.mean(torch.abs(result - ground))
                    def smape(result, ground):
                        numerator = 2 * torch.abs(result - ground)
                        denominator = torch.abs(result) + torch.abs(ground)
                        smape_values = numerator / denominator
                        smape_values[denominator == 0] = 0
                        return torch.mean(smape_values)
                    smape_loss+= groundtruth.size(0) * smape(result, ground)

        mse = squared_loss / total_number

        if mode =='Validating':
            return mse, None, None
        if mode == 'Testing':
            mae=absolute_loss/ total_number
            smape=smape_loss/ total_number
            return mse, mae,smape
