import os
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

logging.getLogger('matplotlib').setLevel(logging.WARNING)

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join(
    "test",
    args.save_dir,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
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

######################################### IMAGE PREPROCESSING #########################################

base_translation_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)
resized_transform = transforms.Compose(
    [
        transforms.Resize(args.GAN_resize),
        base_translation_transform,

    ]
)


# Load and preprocess the image
image_path = '/content/drive/MyDrive/images/sat_512.png'
image = cv2.imread(image_path)
image = cv2.cvtColor((image), cv2.COLOR_BGR2RGB)

image_tensor = resized_transform(Image.fromarray(image))
# Add batch dimension and move to device
image_tensor = image_tensor.unsqueeze(0).to(args.device)

######################################### GENERATING THERMAL IMAGE #########################################

model.netG = model.netG.eval()
with torch.no_grad():
    model.set_input(image_tensor, image_tensor)  
    model.forward() 

    output = model.fake_B
    output = torch.clamp(output, min=-1, max=1)  
    output_images = output * 0.5 + 0.5  

    generated_query = transforms.Grayscale(num_output_channels=3)(
        transforms.Resize(args.GAN_resize)(transforms.ToPILImage()(output_images.squeeze().cpu()))

    )

######################################### SAVING THERMAL IMAGE #########################################
    output_image_path = '/content/drive/MyDrive/images/sat_512_thermal.png'
    generated_query.save(output_image_path)
    print(f"Thermal image saved at: {output_image_path}")

# import os
# import torch
# import numpy as np
# from PIL import Image
# import cv2
# from torchvision import transforms
# import parser
# import commons
# import util
# from model import network

# # تنظیمات اولیه
# args = parser.parse_arguments()
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# commons.make_deterministic(args.seed)

# model = network.pix2pix(args, 3, 1)
# if args.resume is not None:
#     print("loading model ....")
#     model = util.resume_model_pix2pix(args, model)
# model.setup()
# model.netG = model.netG.eval()

# # 
# to_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=0.5, std=0.5),
# ])

# resize = transforms.Resize(args.GAN_resize)

# # مسیر تصویر ورودی
# image_path = '/content/drive/MyDrive/golbahar/frame_01194.png'
# image = cv2.imread(image_path)
# if image is None:
#     raise FileNotFoundError(f"❌ تصویر یافت نشد: {image_path}")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_pil = Image.fromarray(image)

# # برش دو ناحیه ۱۰۲۴×۱۰۲۴ از چپ و راست
# crop_size = 1024
# left_crop = image_pil.crop((0, 0, crop_size, crop_size))
# right_crop = image_pil.crop((image_pil.width - crop_size, 0, image_pil.width, crop_size))

# # تابع پردازش تصویر
# def process_crop(crop):
#     tensor = resize(to_tensor(crop)).unsqueeze(0).to(args.device)
#     with torch.no_grad():
#         model.set_input(tensor, tensor)
#         model.forward()
#         out = model.fake_B.squeeze().cpu().numpy()
#         out = np.clip(out, -1, 1)
#         out = (out + 1) / 2  # to [0,1]
#         if out.ndim == 2:
#             out = np.stack([out]*3, axis=-1)  # (H, W) -> (H, W, 3)
#         return out

# # پردازش هر برش
# left_output = process_crop(left_crop)
# right_output = process_crop(right_crop)

# # ترکیب دو خروجی با میانگین درترک یکسل overlap)
# overlap = 128
# combined = np.zeros((crop_size, crop_size*2 - overlap, 3), dtype=np.float32)

# # قرار دادن قسمت چپ
# combined[:, :crop_size] = left_output

# # قرار دادن قسمت راست
# combined[:, -crop_size:] = right_output

# # # اعمال میانگین در ناحیه مشترک
# # combined[:, crop_size-overlap:crop_size] = (
# #     left_output[:, crop_size-overlap:] * 0.5 +
# #     right_output[:, :overlap] * 0.5
# # )
# for i in range(overlap):
#     alpha = i / (overlap - 1)
#     combined[:, crop_size - overlap + i] = (
#         left_output[:, crop_size - overlap + i] * (1 - alpha) +
#         right_output[:, i] * alpha
#     )
# # ذخیره تصویر نهایی
# final_image = Image.fromarray((combined * 255).astype(np.uint8))
# save_path = '/content/drive/MyDrive/combined_output.png'
# final_image.save(save_path)
# print(f"✅ تصویر نهایی ذخیره شد: {save_path}")