import os
import sys
import torch
import parser
import logging
import h5py
from os.path import join
from datetime import datetime
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

import test
import util
import commons
import datasets_ws
from model import network

# Logging configuration
logging.getLogger('matplotlib').setLevel(logging.WARNING)

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join(
    "test",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.pix2pix(args, 3, 1)

if args.resume is not None:
    print("loading model ....")
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model_pix2pix(args, model)

model.setup()

######################################### LOADING DATASET #########################################
h5_input_path = '//content/drive/MyDrive/STHN/datasets/satellite_0_thermalmapping_135/extended_database.h5'
h5_output_path = '/content/drive/MyDrive/output_dataset_ext.h5'

with h5py.File(h5_input_path, 'r') as h5_input, h5py.File(h5_output_path, 'w') as h5_output:
    images_group = h5_output.create_group("thermal_images")

    for idx, image in enumerate(h5_input['image_data']):
        # Prepare image tensor
        image = Image.fromarray(image)
        image = image.resize((512, 512))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(args.device)

        # Generating thermal image
        empty_target = torch.zeros_like(image_tensor)
        model.set_input(image_tensor, empty_target)
        model.forward()

        output = model.fake_B
        output = torch.clamp(output, min=-1, max=1)  
        output_images = output * 0.5 + 0.5

        generated_query = transforms.Grayscale(num_output_channels=3)(
            transforms.Resize((512, 512))(transforms.ToPILImage()(output_images.squeeze().cpu()))
        )

        # Save generated thermal image
        images_group.create_dataset(str(idx), data=np.array(generated_query))

        print(f"Processed image {idx + 1}/{len(h5_input['image_data'])}")

print(f"Thermal images saved in: {h5_output_path}")
