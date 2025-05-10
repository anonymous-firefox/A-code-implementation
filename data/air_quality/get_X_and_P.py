'''
process csv file
get groundtruth P and rbf result X
'''

import os
os.sched_setaffinity(0, set(range(81)))

cpu_count = os.cpu_count()
print(f"CPU: {cpu_count}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]



def process_file(file_name):
    df = pd.read_csv(file_name)

    df['Date Local'] = pd.to_datetime(df['Date Local'] + ' ' + df['Time Local'], format='%Y-%m-%d %H:%M')
    df = df[df['Date Local'].dt.year == 2018]
    before_interpolation = []
    after_interpolation = []

    time_stamps = pd.date_range('2018-01-01 00:00', '2018-12-31 23:00', freq='H')

    for timestamp in tqdm(time_stamps):

        df_filtered = df[df['Date Local'] == timestamp]


        if df_filtered.empty:
            print(f"Warning: No data for {timestamp} in {file_name}, setting grid to NaN.")
            before_interpolation.append(np.full((len(lat_bins), len(lon_bins)), np.nan))
            after_interpolation.append(np.full((len(lat_bins), len(lon_bins)), np.nan))
            continue


        df_filtered = df_filtered[['Latitude', 'Longitude', 'Sample Measurement']]


        df_filtered = df_filtered[(df_filtered['Latitude'] >= 20) & (df_filtered['Latitude'] <= 55) &
                                  (df_filtered['Longitude'] >= -130) & (df_filtered['Longitude'] <= -60)]


        lat_bins = np.arange(20, 55, 0.5)
        lon_bins = np.arange(-130, -60, 0.5)
        lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)


        df_filtered['Lat_bin'] = (df_filtered['Latitude'] // 0.5) * 0.5
        df_filtered['Lon_bin'] = (df_filtered['Longitude'] // 0.5) * 0.5


        grid_avg = df_filtered.groupby(['Lat_bin', 'Lon_bin'])['Sample Measurement'].mean().reset_index()


        data_grid = np.full(lon_grid.shape, np.nan)
        for _, row in grid_avg.iterrows():
            lat_idx = int((row['Lat_bin'] - 20) / 0.5)
            lon_idx = int((row['Lon_bin'] + 130) / 0.5)
            data_grid[lat_idx, lon_idx] = row['Sample Measurement']

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        #
        # im1 = axes[0].pcolormesh(lon_grid, lat_grid, data_grid, shading='auto', cmap='jet')
        # axes[0].set_title("Before RBF Interpolation")
        # axes[0].set_xlabel("Longitude")
        # axes[0].set_ylabel("Latitude")
        # fig.colorbar(im1, ax=axes[0])


        before_interpolation.append(data_grid)


        valid_points = grid_avg[['Lon_bin', 'Lat_bin']].values
        valid_values = grid_avg['Sample Measurement'].values
        invalid_points = np.argwhere(np.isnan(data_grid))
        if len(invalid_points) > 0:
            interp = RBFInterpolator(valid_points, valid_values)
            interp_points = np.array(
                [[-130 + lon_idx * 0.5, 20 + lat_idx * 0.5] for lat_idx, lon_idx in invalid_points])
            interpolated_values = interp(interp_points)
            for (lat_idx, lon_idx), value in zip(invalid_points, interpolated_values):
                data_grid[lat_idx, lon_idx] = value

        # im2 = axes[1].pcolormesh(lon_grid, lat_grid, data_grid, shading='auto', cmap='jet')
        # axes[1].set_title("After RBF Interpolation")
        # axes[1].set_xlabel("Longitude")
        # fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()
        #

        after_interpolation.append(data_grid)


    return before_interpolation, after_interpolation

for file_name in csv_files:
    print(f"Processing {file_name}...")
    before, after = process_file(file_name)

    np.save(f'{file_name}_truebefore_interpolation.npy', np.array(before))
    np.save(f'{file_name}_after_interpolation.npy', np.array(after))


    print(f"Finished processing {file_name}.")
