import os
import sys
sys.path.append(os.path.join(".."))

import torch
import numpy as np
import tifffile
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from new_model import *
from transforms import *
from utils import *
from dataset import *

def load(checkpoints_dir, model, epoch=1, optimizer=None, device='cpu'):

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  

    checkpoint_path = os.path.join(checkpoints_dir, f'best_model.pth')
    dict_net = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(dict_net['model'])
    optimizer.load_state_dict(dict_net['optimizer'])
    epoch = dict_net['epoch']

    model.to(device)

    print('Loaded %dth network' % epoch)

    return model, epoch

def main():

    #********************************************************#

    project_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\projects\3DN2N-OCM\droso-test_1"
    data_dir = r"\\tier2.embl.de\prevedel\members\Rauscher\data\big_data_small-test\droso_good"
    inference_name = 'inference-3'

    # Extract project_name and method_name
    project_name = os.path.basename(project_dir)
    method_name = os.path.basename(os.path.dirname(project_dir))

    #********************************************************#

    results_dir = os.path.join(project_dir, 'results')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')

    # Make a folder to store the inference
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)
    
    ## Load image stack for inference
    filenames = glob.glob(os.path.join(data_dir, "*.TIFF"))
    print("Following file will be denoised:  ", filenames[0])

    # check if GPU is accessible
    if torch.cuda.is_available():
        print("\nGPU will be used.")
        device = torch.device("cuda:0")
    else:
        print("\nCPU will be used.")
        device = torch.device("cpu")

    mean, std = load_normalization_params(checkpoints_dir)
    
    inf_transform = transforms.Compose([
        NormalizeInference(mean, std),
        CropToMultipleOf16Inference(),
        ToTensorInference(),
    ])

    inv_inf_transform = transforms.Compose([
        ToNumpy(),
        Denormalize(mean, std)
    ])

    inf_dataset = InferenceDataset(
        data_dir,
        transform=inf_transform
    )

    batch_size = 2
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model = UNet3D()
    model, epoch = load(checkpoints_dir, model)

    num_inf = len(inf_dataset)
    num_batch = int((num_inf / batch_size) + ((num_inf % batch_size) != 0))

    print("starting inference")
    output_images = []  # List to collect output images

    with torch.no_grad():
        model.eval()

        for batch, data in enumerate(inf_loader):
            input_stack = data.to(device)  # Assuming data is already a tensor of the right shape

            output_stack = model(input_stack)

            for i in range(0, batch_size):
                output_stack_pt = output_stack[i, 0, :, :, :]
                output_stack_np = inv_inf_transform(output_stack_pt)  # Convert output tensors to numpy format for saving

                output_images.append(output_stack_np)

            print('BATCH %04d/%04d' % (batch, len(inf_loader)))
    
    # Stack and save output images
    output_stack = np.stack(output_images, axis=0)
    filename = f'output_stack-{method_name}-{project_name}-{inference_name}-epoch{epoch}.TIFF'
    tifffile.imwrite(os.path.join(inference_folder, filename), output_stack)

    print("TIFF stacks created successfully.")

if __name__ == '__main__':
    main()

