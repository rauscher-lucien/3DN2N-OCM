import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_stack, target_stack = data

        # Normalize input image
        input_normalized = (input_stack - self.mean) / self.std

        # Normalize target image
        target_normalized = (target_stack - self.mean) / self.std

        return input_normalized, target_normalized
    

class NormalizeInference(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_stack = data

        # Normalize input image
        input_normalized = (input_stack - self.mean) / self.std

        return input_normalized


class RandomFlip(object):

    def __call__(self, data):

        input_img, target_img = data

        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)

        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)

        return input_img, target_img
    

class RandomHorizontalFlip:
    def __call__(self, data):
        """
        Apply random horizontal flipping to both the input stack of slices and the target slice.
        In 50% of the cases, only horizontal flipping is applied without vertical flipping.
        
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        
        Returns:
            Tuple: Horizontally flipped input stack and target slice, if applied.
        """
        input_stack, target_slice = data

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            # Flip along the width axis (axis 1), keeping the channel dimension (axis 2) intact
            input_stack = np.flip(input_stack, axis=1)
            target_slice = np.flip(target_slice, axis=1)

        # With the modified requirements, we remove the vertical flipping part
        # to ensure that only horizontal flipping is considered.

        return input_stack, target_slice



class RandomCrop:
    def __init__(self, output_size=(64, 64)):
        """
        RandomCrop constructor for cropping both the input stack of slices and the target slice.
        Args:
            output_size (tuple): The desired output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply the cropping operation.
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        Returns:
            Tuple: Cropped input stack and target slice.
        """
        input_stack, target_stack = data

        _, h, w = input_stack.shape
        _, h, w = target_stack.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input_cropped = input_stack[:, top:top+new_h, left:left+new_w]
        target_cropped = target_stack[:, top:top+new_h, left:left+new_w]

        return (input_cropped, target_cropped)



class CropToMultipleOf16Inference(object):
    """
    Crop each stack of images to ensure their depth, height, and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, stack):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (depth, height, width).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """
        d, h, w = stack.shape  # Assuming stack is a numpy array with shape (depth, height, width)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the stack
        cropped_stack = stack[:, top:top + new_h, left:left + new_w]

        return cropped_stack



class ToTensor(object):
    def __call__(self, data):
        def convert_image(img):
            return torch.from_numpy(img.astype(np.float32))
        return tuple(convert_image(img) for img in data)
    

class ToTensorInference(object):
    def __call__(self, data):
        return torch.from_numpy(data.astype(np.float32))





class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy()
    
    
    
    
class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor


class ToTensorVideo(object):
    """
    Convert stacks of images or single images to PyTorch tensors, specifically handling a tuple
    of a stack of input frames and a single target frame for grayscale images.
    The input stack is expected to have the shape (4, H, W, 1) and the target image (H, W, 1).
    It converts them to PyTorch's (B, C, H, W) format for the stack and (C, H, W) for the single image.
    """

    def __call__(self, data):
        """
        Convert a tuple of input stack and target image to PyTorch tensors, adjusting the channel position.
        
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).
        
        Returns:
            tuple of torch.Tensor: A tuple containing the converted input stack as a PyTorch tensor
                                   with shape (4, 1, H, W) and the target image as a PyTorch tensor
                                   with shape (1, H, W).
        """
        input_stack, target_img = data
        
        # Convert the input stack to tensor and adjust dimensions
        input_stack_tensor = torch.from_numpy(input_stack.transpose(3, 1, 2, 0).astype(np.float32))
        
        # Convert the target image to tensor and adjust dimensions
        target_img_tensor = torch.from_numpy(target_img.transpose(2, 0, 1).astype(np.float32))
        
        return input_stack_tensor, target_img_tensor
    

class ToNumpyVideo(object):
    """
    Convert PyTorch tensors to numpy arrays, handling single images, batches of images, 
    and stacks of images separately. Adjusts dimensions to move the channel dimension to the last.
    """
    def __call__(self, tensor):
        # Convert a single image or a batch of images to a numpy array
        if tensor.ndim == 4:  # Single image or batch of images (B, C, H, W)
            return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        elif tensor.ndim == 5:  # Stack of images (B, C, H, W, Num_Frames)
            return tensor.to('cpu').detach().numpy().transpose(0, 2, 3, 1, 4)
        else:
            raise ValueError("Unsupported tensor format: input must be a single image (C, H, W), \
                             a batch of images (B, C, H, W), or a stack of images (B, C, H, W, Num_Frames).")
        


class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)
        
        return img_16bit