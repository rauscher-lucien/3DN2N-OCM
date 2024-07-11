import os
import numpy as np
import torch
import glob
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class VolumeSubstackDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, stack_depth=32, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.stack_depth = stack_depth
        self.preloaded_data = {}
        self.pairs = self.preload_volumes(root_folder_path)

    def preload_volumes(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume
                num_slices = volume.shape[0]
                # Store indices for the start of each possible substack
                total_possible_stacks = (num_slices - 2 * self.stack_depth) // 2 + 1
                for i in range(total_possible_stacks):
                    start_index = 2 * i  # Start index of the first slice in the substack
                    pairs.append((full_path, start_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, start_index = self.pairs[index]
        
        # Access the preloaded entire volume
        volume = self.preloaded_data[file_path]
        
        # Determine indices for the input and target substacks
        input_indices = range(start_index, start_index + 2 * self.stack_depth, 2)
        target_indices = range(start_index + 1, start_index + 1 + 2 * self.stack_depth, 2)
        
        # Fetch the actual slices
        input_stack = volume[input_indices]
        target_stack = volume[target_indices]

        if self.transform:
            input_stack, target_stack = self.transform((input_stack, target_stack))

        input_stack = input_stack[np.newaxis, ...]
        target_stack = target_stack[np.newaxis, ...]

        return input_stack, target_stack





class FinalDatasetExtraNoise(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, num_adjacent, transform=None, std_dev_range=(0.01, 0.05), noise_shift_range=(-0.1, 0.1)):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.num_adjacent = num_adjacent  # Number of slices adjacent to the central slice
        self.std_dev_range = std_dev_range  # Range for the standard deviation of noise
        self.noise_shift_range = noise_shift_range  # Range for the noise shift
        self.preloaded_data = {}  # To store preloaded data
        self.pairs, self.cumulative_slices = self.preload_and_make_pairs(root_folder_path)

    def preload_and_make_pairs(self, root_folder_path):
        pairs = []
        cumulative_slices = [0]
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume  # Preload data here
                num_slices = volume.shape[0]
                if num_slices > 2 * self.num_adjacent:  # Ensure enough slices for forming pairs
                    for i in range(self.num_adjacent, num_slices - self.num_adjacent):
                        input_slices_indices = list(range(i - self.num_adjacent, i)) + list(range(i + 1, i + 1 + self.num_adjacent))
                        target_slice_index = i
                        pairs.append((full_path, input_slices_indices, target_slice_index))
                        cumulative_slices.append(cumulative_slices[-1] + 1)
        return pairs, cumulative_slices

    def add_gaussian_noise(self, image, std_dev, mean_shift):
        noise = np.random.normal(mean_shift, std_dev, image.shape)
        return image + noise

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        file_path, input_slice_indices, target_slice_index = self.pairs[pair_index]

        volume = self.preloaded_data[file_path]
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)
        target_slice = volume[target_slice_index][..., np.newaxis]

        # Sample standard deviation and mean shift once per call
        std_dev = np.random.uniform(*self.std_dev_range) * np.max(volume)
        mean_shift = np.random.uniform(*self.noise_shift_range) * np.max(volume)

        # Add noise independently to each slice and target
        input_slices = np.stack([self.add_gaussian_noise(volume[i], std_dev, mean_shift) for i in input_slice_indices], axis=-1)
        target_slice = self.add_gaussian_noise(target_slice, std_dev, mean_shift)

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice
    

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, stack_depth=32, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.stack_depth = stack_depth
        self.preloaded_data = {}
        self.pairs = self.preload_volumes(root_folder_path)

    def preload_volumes(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume
                num_slices = volume.shape[0]
                # Store indices for the start of each possible substack
                total_possible_stacks = (num_slices - 2 * self.stack_depth) // 2 + 1
                for i in range(total_possible_stacks):
                    start_index = 2 * i  # Start index of the first slice in the substack
                    pairs.append((full_path, start_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, start_index = self.pairs[index]
        
        # Access the preloaded entire volume
        volume = self.preloaded_data[file_path]
        
        # Determine indices for the input and target substacks
        input_indices = range(start_index, start_index + 2 * self.stack_depth, 2)
        target_indices = range(start_index + 1, start_index + 1 + 2 * self.stack_depth, 2)
        
        # Fetch the actual slices
        input_stack = volume[input_indices]
        target_stack = volume[target_indices]

        if self.transform:
            input_stack = self.transform((input_stack, target_stack))

        input_stack = input_stack[np.newaxis, ...]

        return input_stack