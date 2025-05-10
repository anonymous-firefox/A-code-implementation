import os
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from cfg import cfg
import matplotlib

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings

warnings.filterwarnings('ignore')


def find_available(input_indices, length=1):
    if length == 1:
        return input_indices

    indices = []

    for i in range(len(input_indices) - length + 1):
        if input_indices[i + length - 1] - input_indices[i] == length - 1:
            indices.append(input_indices[i])
    return indices


class PollutionDataset(Dataset):
    def __init__(self, data_path, if_all=0):
        self.device=cfg.device
        if cfg.dataset=='CHN':
            filetail='.npy'
        elif cfg.dataset=='USA':
            filetail = '.csv_after_interpolation.npy'
        if if_all:
            npy_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(filetail)])
            tensors = [torch.tensor(np.load(file), device=cfg.device, dtype=torch.float) for file in npy_files]
            self.data = torch.stack(tensors, dim=1)
        else:
            self.data = torch.tensor(np.load(data_path), device=cfg.device, dtype=torch.float).unsqueeze(1)

        self.max_values = self._compute_max_values().to('cpu')
        self.num_samples = len(self.max_values)

    def _compute_max_values(self):

        nan_mask = torch.isnan(self.data)
        neg_inf_tensor = torch.full_like(self.data, float(0), device=self.device)
        data_without_nan = torch.where(nan_mask, neg_inf_tensor, self.data)

        max_values = [torch.max(data_without_nan[:3], dim=0, keepdim=True).values]
        for start in range(3, self.data.shape[0], 6):
            end = start + 6
            if end <= self.data.shape[0]:
                segment = data_without_nan[start:end]
                max_segment = torch.max(segment, dim=0, keepdim=True).values


                if torch.all(nan_mask[start:end]):
                    max_segment[:] = float('nan')

                max_values.append(max_segment)

        return torch.cat(max_values, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.max_values[idx]


class AtmosphereDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.dates = pd.date_range(start='2018-01-01', end='2018-12-31', freq='D')
        self.hours = ['00', '06', '12', '18']
        self.file_list = [
            f'gfs_4_{date.strftime("%Y%m%d")}_{hour}00_000.npz'
            for date in self.dates for hour in self.hours
        ]

        # Define the preload fraction (cfg.preload ranges from 0 to 1)
        self.preload_fraction = cfg.preload
        self.num_files = len(self.file_list)

        # Calculate how many files to preload
        self.num_to_preload = int(self.num_files * self.preload_fraction)

        # If cfg.preload > 0, preload a fraction of the files
        if self.preload_fraction > 0:
            self.preloaded_data = [None] * self.num_files  # Placeholder for preloaded data

            # Preload a fraction of the dataset with a progress bar
            for idx in tqdm(range(self.num_to_preload), desc="Preloading files", unit="file"):
                filename = self.file_list[idx]
                file_path = os.path.join(self.data_path, filename)

                if os.path.exists(file_path):
                    with np.load(file_path) as data:
                        # Preload and stack all the arrays into a list
                        all_arrays = [torch.tensor(data[key], dtype=torch.float32) for key in
                                      data.keys()]
                        self.preloaded_data[idx] = torch.stack(all_arrays)  # Shape: (num_keys, 80, 130)
                else:
                    # If the file does not exist, store a tensor of zeros
                    self.preloaded_data[idx] = torch.zeros(cfg.atmosphere_size, dtype=torch.float32)
        else:
            self.preloaded_data = None  # No preloading, load on demand

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # If data is preloaded for this index, return the preloaded data
        if self.preloaded_data and self.preloaded_data[idx] is not None:
            return self.preloaded_data[idx]

        # Otherwise, load the data on demand
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_path, filename)

        if os.path.exists(file_path):
            with np.load(file_path) as data:
                # Extract all arrays and convert them to PyTorch tensors
                all_arrays = [torch.tensor(data[key], dtype=torch.float32) for key in data.keys()]
                return torch.stack(all_arrays)  # Stack them into a single tensor
        else:
            # If file doesn't exist, return a tensor of zeros
            return torch.zeros(cfg.atmosphere_size, dtype=torch.float32)


import torch
from torch.utils.data import Dataset


class GroundTruthDataset(Dataset):
    def __init__(self, data_path,if_all=0):
        if cfg.dataset=='CHN':
            filetail='.npy'
        elif cfg.dataset=='USA':
            filetail = '.csv_truebefore_interpolation.npy'
        if if_all:
            npy_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(filetail)])
            tensors = [torch.tensor(np.load(file), device=cfg.device, dtype=torch.float) for file in npy_files]
            self.data = torch.stack(tensors, dim=1)
        else:
            self.data = torch.tensor(np.load(data_path), device=cfg.device, dtype=torch.float).unsqueeze(1)
        self.aggregated_values = self._compute_aggregated_values().to('cpu')
        self.num_samples = len(self.aggregated_values)

    def _compute_aggregated_values(self):
        aggregated_values = []

        nan_mask = torch.isnan(self.data)
        data_no_nan = torch.where(nan_mask, torch.full_like(self.data, float(0)), self.data)

        aggregated_values.append(torch.max(data_no_nan[:3], dim=0, keepdim=True).values)

        for start in range(3, self.data.shape[0], 6):
            end = start + 6
            if end <= self.data.shape[0]:
                segment = data_no_nan[start:end]
                max_segment = torch.max(segment, dim=0, keepdim=True).values

                if torch.all(nan_mask[start:end]):
                    max_segment[:] = float('nan')
                aggregated_values.append(max_segment)

        result = torch.cat(aggregated_values, dim=0)

        result = torch.where(result == float(0), torch.full_like(result, float('nan')), result)
        return result

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):


        return self.aggregated_values[idx]


class Combined_sequence_Dataset(Dataset):
    def __init__(self, atmosphere_dataset, pollution_dataset=None, groundtruth_dataset=None, time_length=2):
        self.atmosphere_dataset = atmosphere_dataset
        self.pollution_dataset = pollution_dataset
        self.groundtruth_dataset = groundtruth_dataset
        self.time_length = time_length
        assert len(atmosphere_dataset) == len(pollution_dataset), "Datasets must be of the same length"

    def __len__(self):
        # Adjust the length to return only full sequences
        return len(self.atmosphere_dataset) - self.time_length + 1

    def __getitem__(self, idx):
        if idx + self.time_length > len(self.atmosphere_dataset):

            raise IndexError("Index out of bounds for the requested sequence length")


        atmosphere_sequence = [self.atmosphere_dataset[idx + i] for i in range(self.time_length)]
        pollution_sequence = [self.pollution_dataset[idx + i] for i in range(self.time_length)] if self.pollution_dataset else None
        groundtruth_sequence = [self.groundtruth_dataset[idx + i] for i in range(self.time_length)] if self.groundtruth_dataset else None


        atmosphere_tensor = torch.stack(atmosphere_sequence)
        pollution_tensor = torch.stack(pollution_sequence) if pollution_sequence else None
        groundtruth_tensor = torch.stack(groundtruth_sequence) if groundtruth_sequence else None
        if cfg.dataset=='USA' and atmosphere_tensor.shape[2] != cfg.atmosphere_size[1]:
            atmosphere_tensor=atmosphere_tensor[:,:,:-1,:-1]


        if atmosphere_tensor.shape[1:] != cfg.atmosphere_size:
            print(atmosphere_tensor.shape,cfg.atmosphere_size)
            raise ValueError("Atmosphere data shape mismatch")

        if pollution_tensor is not None and pollution_tensor.shape[2:] != cfg.map_size:
            raise ValueError(f"Pollution data shape mismatch {pollution_tensor.shape}")

        if cfg.dataset=='USA':

            pollution_tensor = torch.flip(pollution_tensor, [2])
            groundtruth_tensor = torch.flip(groundtruth_tensor, [2])
        return atmosphere_tensor, pollution_tensor, groundtruth_tensor


current_working_directory = os.getcwd()
if current_working_directory.split('/')[-1] == 'dataset':
    dir_ = '../'
else:
    dir_ = ''
    dir_=cfg.datadir



if cfg.dataset=='CHN':
    if cfg.pollution_type!='all':
        pollution_dataset = PollutionDataset(dir_ + f'data/air_quality/processed_npy/{cfg.pollution_type}_interpolated_map.npy')
    else:
        pollution_dataset = PollutionDataset(
            dir_ + f'data/air_quality/processed_npy', if_all=1)
    #
    atmosphere_dataset = AtmosphereDataset(dir_ + 'data/atmosphere')

    if cfg.pollution_type!='all':
        groundtruth_dataset = GroundTruthDataset(
        dir_ + f'data/air_quality/{cfg.pollution_type}_interpolated_map.npy')
    else:
        groundtruth_dataset = GroundTruthDataset(
            dir_ + f'data/air_quality',if_all=1)
elif cfg.dataset=='USA':
    if cfg.pollution_type!='all':
        pollution_dataset = PollutionDataset(dir_ + f'dataUSA/air_quality/{cfg.pollution_type}.csv_after_interpolation.npy')
    else:
        pollution_dataset = PollutionDataset(dir_ + f'dataUSA/air_quality', if_all=1)
    #
    atmosphere_dataset = AtmosphereDataset(dir_ + 'dataUSA/atmosphere')


    if cfg.pollution_type!='all':
        groundtruth_dataset = GroundTruthDataset(
        dir_ + f'dataUSA/air_quality/{cfg.pollution_type}.csv_truebefore_interpolation.npy')
    else:
        groundtruth_dataset = GroundTruthDataset(
            dir_ + f'dataUSA/air_quality',if_all=1)

combined_dataset = Combined_sequence_Dataset(atmosphere_dataset, pollution_dataset, groundtruth_dataset, cfg.length)
num_workers=cfg.num_workers

pt = dir_ + f'dataset/saved/available_1_indices_.pt'
if cfg.dataset=='USA':
    pt = f'dataset/saved/available_1_indices_.pt'
else:
    pt = dir_ + f'dataset/saved/available_1_indices_.pt'

loaded_indices = torch.load(pt, weights_only=True) #load valid sample


loaded_indices = find_available(loaded_indices, length=cfg.length)
if cfg.test:
    loaded_indices=loaded_indices[:int(0.01*len(loaded_indices))]
    num_workers=0
elif cfg.light:
    loaded_indices=loaded_indices[:int(0.3*len(loaded_indices))]

combined_dataset = Subset(combined_dataset, loaded_indices)



total_size = len(combined_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
print('total valid sample:', total_size)
generator = torch.Generator().manual_seed(cfg.seed)
train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size],
                                                        generator=generator)

batch=cfg.batch_size

pin_memory=True
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=pin_memory,num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, pin_memory=pin_memory,num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, pin_memory=pin_memory,num_workers=num_workers)



if __name__ == '__main__':
    for atmosphere, pollution, groundtruth in train_loader:
        print(
            f'idx:,Train Batch - Inputs shape: {atmosphere.shape}, Targets shape: {pollution.shape}, Targets shape: {groundtruth.shape}')

        for t in range(16):
            atmosphere_2d = atmosphere[0, t, 250, :, :].to('cpu')
            pollution_2d = pollution[0, t, 0, :, :].to('cpu')
            groundtruth_2d = groundtruth[0, t, 0, :, :].to('cpu')

            def save_plot_2d_data(data, title, filename):
                plt.figure(figsize=(8, 6))

                im = plt.imshow(data, cmap='viridis', interpolation='nearest', vmin=0)
                plt.colorbar(label='Value', shrink=0.55)
                plt.title(title)
                plt.xlabel('Longitude Index')
                plt.ylabel('Latitude Index')
                plt.savefig(filename, dpi=1200)
                plt.close()

            save_plot_2d_data(atmosphere_2d, "Atmosphere (2D Slice)", f"img2/atmosphere_sliceUSA{t}.png")
            save_plot_2d_data(pollution_2d, f"Pollution (2D Slice)", f"img2/pollution_sliceUSA{t}.png")
            save_plot_2d_data(groundtruth_2d,
                              "PM2.5 Ground Truth at a specific time in the United States",
                              f"img2/groundtruth_sliceUSA{t}.png")

       
        break
