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

    project_dir = r"C:\Users\rausc\Documents\EMBL\projects\3DN2N-OCM\OCT-data-1-test_1"
    data_dir = r"C:\Users\rausc\Documents\EMBL\data\big_data_small\OCT-data-1"
    inference_name = 'inference'


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
        Normalize(mean, std),
        CropToMultipleOf32Inference(),
        ToTensorInference(),
    ])

    inv_inf_transform = transforms.Compose([
        BackTo01Range(),
        ToNumpy()
    ])

    inf_dataset = InferenceDataset(
        data_dir,
        transform=inf_transform
    )

    batch_size = 1
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    
    model = UNet3D()
    model, _ = load(checkpoints_dir, model)

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
                output_img = output_stack[i, 0, 0, :, :]
                output_img_np = inv_inf_transform(output_img)  # Convert output tensors to numpy format for saving

                output_images.append(output_img_np)

            print('BATCH %04d/%04d' % (batch, len(inf_loader)))

    # Clip output images to the 0-1 range
    output_images_clipped = [np.clip(img, 0, 1) for img in output_images]
    
    # Stack and save output images
    output_stack = np.stack(output_images_clipped, axis=0)
    tifffile.imwrite(os.path.join(inference_folder, 'output_stack.TIFF'), output_stack)

    print("TIFF stacks created successfully.")

if __name__ == '__main__':
    main()



#     print("starting inference")
#     with torch.no_grad():

#         netG.eval()

#         for batch, data in enumerate(inf_loader):

#             input_img = data[0].to(device)
#             output_img = netG(input_img)

#             input_img = inv_inf_transform(input_img)[..., 0]
#             output_img = inv_inf_transform(output_img)[..., 0]

#             # input_img = np.clip(input_img, 0, 1)
#             # output_img = np.clip(output_img, 0, 1)

#             for j in range(0, batch_size):
#                 name1 = batch
#                 name2 = j
#                 fileset = {'name': name,
#                             'input': "%04d-%04d-input.png" % (name1, name2),
#                             'output': "%04d-%04d-output.png" % (name1, name2),
#                             'target': "%04d-%04d-label.png" % (name1, name2)}

#                 input_img_1 = np.squeeze(input_img[j, :, :])
#                 output_img_1 = np.squeeze(output_img[j, :, :])

#                 input_img_1 = (input_img_1 * 255).astype(np.uint8)
#                 output_img_1 = (output_img_1 * 255).astype(np.uint8)

#                 input_img_path = os.path.join(inference_folder, fileset['input'])
#                 output_img_path = os.path.join(inference_folder, fileset['output'])

#                 # Image.fromarray(input_img_1).save(input_img_path)
#                 Image.fromarray(output_img_1).save(output_img_path)


# if __name__ == '__main__':
#     main()

