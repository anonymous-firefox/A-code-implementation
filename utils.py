import os
import re
import torch
from cfg import cfg
import subprocess
import time



class Apply_Normalization():
    def __init__(self, norm_params):
        self.device = cfg.device
        self.norm_params = norm_params
        #
        # if cfg.dataset == 'CHN':
        #     self.atmosphere_mean = self.norm_params['atmosphere_mean'].to(self.device)  # Shape: [channels]
        #     self.atmosphere_std = self.norm_params['atmosphere_std'].to(self.device)  # Shape: [channels]
        #     self.groundtruth_mean = torch.tensor([67.0471, 0.8425, 27.4507, 65.0885, 75.4055, 39.2343, 13.7257],device=self.device)
        #     self.groundtruth_std = torch.tensor([51.4647, 0.5298, 20.5887, 43.9906, 96.2521, 38.5007, 16.2260],device=self.device)
        # elif cfg.dataset == 'USA':
        #     self.groundtruth_mean = torch.tensor([0.2688, 6.6581, 6.2562, 0.0311, 20.9838, 8.5005, 0.6970],device=self.device)
        #     self.groundtruth_std = torch.tensor([0.2068, 7.5112, 12.1883, 0.0149, 35.1267, 11.1296, 2.7963],device=self.device)
        #     self.atmosphere_mean =torch.tensor([],device=self.device)
        #     self.atmosphere_std = torch.tensor([],device=self.device)

        if cfg.pollution_type in cfg.pollution_list:
            index = cfg.pollution_list.index(cfg.pollution_type)
            self.groundtruth_mean =self.groundtruth_mean.clone().detach()[index].view(1)
            self.groundtruth_std = self.groundtruth_std.clone().detach()[index].view(1)

        self.pollution_mean = self.groundtruth_mean
        self.pollution_std = self.groundtruth_std

    def __call__(self, atmosphere, pollution, groundtruth):

        atmosphere = atmosphere.to(self.device)
        pollution = pollution.to(self.device)
        groundtruth = groundtruth.to(self.device)
        if self.norm_params is None and cfg.dataset == 'CHN':
            print('Norm_params File Not Found')
            return atmosphere, pollution, groundtruth


        with torch.no_grad():
            if atmosphere.ndim == 5:  # [batch, time_steps, channels, H, W]
                atmosphere_mean = self.atmosphere_mean[None, None, :, None, None]
                atmosphere_std = self.atmosphere_std[None, None, :, None, None]
            else:
                raise ValueError(f"Unsupported atmosphere shape: {atmosphere.shape}")


            if pollution.ndim == 5:  # [batch, time_steps, channels, H, W]
                pollution_mean = self.pollution_mean[None, None, :, None, None]
                pollution_std = self.pollution_std[None, None, :, None, None]
            else:
                raise ValueError(f"Unsupported pollution shape: {pollution.shape}")

            if groundtruth.ndim == 5:  # [batch, time_steps, channels, H, W]
                groundtruth_mean = self.groundtruth_mean[None, None, :, None, None]
                groundtruth_std = self.groundtruth_std[None, None, :, None, None]
            else:
                raise ValueError(f"Unsupported pollution shape: {groundtruth.shape}")

            epsilon = 1e-6

            adjusted_atmosphere_std = torch.where(torch.abs(atmosphere_std) < epsilon,
                                                  torch.tensor(1.0, device=self.device, dtype=torch.float),
                                                  atmosphere_std)
            atmosphere = (atmosphere - atmosphere_mean) / adjusted_atmosphere_std

            adjusted_pollution_std = torch.where(torch.abs(pollution_std) < epsilon,
                                                 torch.tensor(1.0, device=self.device, dtype=torch.float),
                                                 pollution_std)
            pollution = (pollution - pollution_mean) / adjusted_pollution_std

            # adjusted_groundtruth_std = torch.where(torch.abs(groundtruth_std) < epsilon, torch.tensor(1.0, device=device),groundtruth_std)
            groundtruth = (groundtruth - groundtruth_mean) / groundtruth_std
            return atmosphere, pollution, groundtruth

    def reverse_normalize(self, inputs):
        mean_expanded = self.groundtruth_mean[None, None, :, None, None]
        std_expanded = self.groundtruth_std[None, None, :, None, None]

        valid_mask = ~torch.isnan(std_expanded)

        outputs = torch.where(
            valid_mask,
            inputs * std_expanded + mean_expanded,
            inputs
        )
        return outputs


def load_normalization_params(file_path):
    if os.path.exists(file_path):
        print(f"Loading normalization: {file_path}")
        return torch.load(file_path, weights_only=True)
    else:
        print(f"No normalization parameters found at {file_path}")
        return None


#  [8, 2, 80, 130]  (8, 4, 1, 80, 130) (8, 4, 1, 80, 130)
def calculate_criterion(outputs, pollution, groundtruth):
    groundtruth = groundtruth[:, cfg.input_length:].squeeze(dim=2)

    if len(outputs.shape)==5 and outputs.shape[0]==1:
        outputs=outputs.squeeze(0)
    if cfg.which_is_target!='all' and cfg.pollution_type=='all':
        index = cfg.pollution_list.index(cfg.which_is_target)
        outputs=outputs[:, :, index:index+1, :, :]
        groundtruth=groundtruth[:, :, index:index+1, :, :]

    mask = ~torch.isnan(outputs) & ~torch.isnan(groundtruth)
    return outputs[mask], groundtruth[mask]



def delete_too_much_file(weights_dir):
    for root, dirs, files in os.walk(weights_dir):
        if root != weights_dir:
            files_with_loss = []
            for file in os.listdir(root):
                if os.path.isfile(os.path.join(root, file)):
                    parts = file.split('_')
                    if len(parts) > 3:
                        try:
                            val_loss_str = float(parts[3])
                            files_with_loss.append((file, val_loss_str))
                        except ValueError:
                            print(f"val_loss_str in file {file} cannot be converted to float, skipping this file")

            files_with_loss.sort(key=lambda x: x[1])


            files_to_delete = [file for file, _ in files_with_loss[4:]]


            delete_count_by_dir = {}

            for file in files_to_delete:
                path = os.path.join(root, file)
                parent_dir = os.path.dirname(path)
                os.remove(path)
                if parent_dir in delete_count_by_dir:
                    delete_count_by_dir[parent_dir] += 1
                else:
                    delete_count_by_dir[parent_dir] = 1

            for parent_dir, count in delete_count_by_dir.items():
                print(f"Deleted {count} files under directory {parent_dir}")





def load_best_model_and_optimizer(model, optimizer=None, weights_dir='weights',forceload=0):

    if not cfg.load_optimizer and not cfg.load_best_weights and not forceload:
        return model, optimizer



    if not cfg.weight_dir:
        weights_dir = 'weights'

    best_model_path = None
    best_optimizer_path = None
    min_val_loss = float('inf')

    if not any(os.path.isdir(os.path.join(weights_dir, item)) for item in os.listdir(weights_dir)):

        for file in os.listdir(weights_dir):
            file_path = os.path.join(weights_dir, file)
            if os.path.isfile(file_path) and file.startswith(cfg.model_type) and file.endswith(
                    '.pth') and not file.endswith('optimizer.pth'):
                try:
                    parts = file.split('_')
                    val_loss_str = parts[3]
                    val_loss = float(val_loss_str)

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_model_path = file_path
                except Exception as e:

                    continue
    else:

        for root, dirs, files in os.walk(weights_dir):
            if root != weights_dir:
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.startswith(cfg.model_type) and file.endswith('.pth') and not file.endswith('optimizer.pth'):
                        try:
                            parts = file.split('_')
                            val_loss_str = parts[3]
                            val_loss = float(val_loss_str)

                            if val_loss < min_val_loss:
                                min_val_loss = val_loss
                                best_model_path = file_path
                        except Exception as e:

                            continue
    if optimizer is not None and best_optimizer_path and cfg.load_optimizer:
        best_optimizer_path = best_model_path.replace('.pth', '_optimizer.pth')

    if best_model_path and cfg.load_best_weights:

        if not cfg.model_type=='TAU3':
            print(f"Loading weights: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path,
                                             weights_only=True,
                                             map_location=torch.device(cfg.device)),
                                  strict=False)
        else:
            print(f"Loading weights (structure partially incompatible): {best_model_path}")
            pretrained_dict = torch.load(best_model_path,
                                                 weights_only=True,
                                                 map_location=torch.device(cfg.device))

            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)




    if optimizer is not None and best_optimizer_path and cfg.load_optimizer:
        optimizer.load_state_dict(torch.load(best_optimizer_path,
                                             weights_only=True,
                                             map_location=torch.device(cfg.device)))
        print(f"Loading optimizer: {best_optimizer_path}")
    return model, optimizer




