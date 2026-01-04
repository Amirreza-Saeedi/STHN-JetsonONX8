import os
import torch
import h5py
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import logging
from model import network
import util, commons
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
import copy
from PIL import Image

import test
import util
import commons
import datasets_ws
from model import network

# added libraries
from torchvision import transforms #, functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# تنظیمات پایه
h5_input_path = '/content/drive/MyDrive/STHN/datasets/satellite_0_thermalmapping_135/val_database.h5'
save_dir = '/content/drive/MyDrive/thermal_val_database11/'
os.makedirs(save_dir, exist_ok=True)

# تنظیمات مدل
args = parser.parse_arguments()
args.save_dir = save_dir
# commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

model = network.pix2pix(args, 3, 1)
if args.resume is not None:
    print("loading model ....")
    model = util.resume_model_pix2pix(args, model)
model.setup()
model.netG = model.netG.eval()

base_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[ 0.406, 0.456,0.485], std=[0.225, 0.224, 0.229]),
    transforms.Normalize(mean=0.5, std=0.5),

])
resized_transform = transforms.Compose([
    transforms.Resize(args.GAN_resize)
])

# بارگذاری دیتاست H5
with h5py.File(h5_input_path, 'r') as h5_file:
    image_data = h5_file['image_data']
    print(f"✅ Total images: {len(image_data)}")

    for idx, img_np in enumerate(image_data):
        
        print(f"🔹 Processing image {idx + 1}/{len(image_data)}...")
        img_np = img_np[..., ::-1]
        image_tensor = resized_transform(base_transform(Image.fromarray(img_np)))
        image_tensor = image_tensor.unsqueeze(0).to(args.device)

        with torch.no_grad():
            model.set_input(image_tensor, image_tensor)
            model.forward()

            output = model.fake_B
            output = torch.clamp(output, min=-1, max=1)  
            output_images = output * 0.5 + 0.5

            generated_image = transforms.ToPILImage()(output_images.squeeze().cpu())
            generated_query = transforms.Grayscale(num_output_channels=3)(
                    transforms.Resize(args.GAN_resize)(transforms.ToPILImage()(output_images.squeeze().cpu()))
            )
            output_image_path = os.path.join(save_dir, f"generated_image_{idx + 1}.png")
            generated_query.save(output_image_path)
            print(f"✅ Saved: {output_image_path}")
            
            
