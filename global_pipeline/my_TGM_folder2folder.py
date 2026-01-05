# import os
# import torch
# import h5py
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# import logging
# from model import network
# import util, commons
# import parser
# folder_name = "crops_1"
# # تنظیمات پایه
# input_dir = f"/content/drive/MyDrive/{folder_name}/small_patches"
# generated_save_dir = f"/content/drive/MyDrive/{folder_name}/thermal"
# os.makedirs(generated_save_dir, exist_ok=True)

# # تنظیمات مدل
# args = parser.parse_arguments()
# commons.make_deterministic(args.seed)

# model = network.pix2pix(args, 3, 1)
# if args.resume is not None:
#     print("loading model ....")
#     model = util.resume_model_pix2pix(args, model)
# model.setup()
# model.netG = model.netG.eval()

# base_translation_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=0.5, std=0.5),
#     ]
# )
# resized_transform = transforms.Compose(
#     [
#         transforms.Resize(args.GAN_resize),
#         base_translation_transform,

#     ]
# )

# # تنظیمات برش
# crop_size = (1024, 1024)
# # بارگذاری و پردازش تصاویر
# for filename in os.listdir(input_dir):
#     if filename.endswith(('.png', '.jpg', '.jpeg','tif')):
#         image_path = os.path.join(input_dir, filename)
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor((image), cv2.COLOR_BGR2RGB)

  
#         # image = Image.open(image_path)
#         # # برش از وسط تصویر
#         # width, height = image.size
#         # left = (width - crop_size[0]) / 2
#         # top = (height - crop_size[1]) / 2
#         # right = left + crop_size[0]
#         # bottom = top + crop_size[1]

#         # cropped_image = image.crop((left, top, right, bottom))
        
#         image_tensor = resized_transform(Image.fromarray(image))
#         image_tensor = image_tensor.unsqueeze(0).to(args.device)

#         with torch.no_grad():
#             model.set_input(image_tensor, image_tensor)
#             model.forward()

#             output = model.fake_B
#             output = torch.clamp(output, min=-1, max=1)  
#             output_images = output * 0.5 + 0.5

#             generated_image = transforms.ToPILImage()(output_images.squeeze().cpu())

#             generated_image_path = os.path.join(generated_save_dir, filename)
#             generated_image.save(generated_image_path)

#         print(f"✅ Cropped and processed image saved: {generated_image_path}")



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

# # تبدیلات
# to_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=0.5, std=0.5),
# ])
# resize = transforms.Resize(args.GAN_resize)

# # مسیر پوشه‌های ورودی و خروجی
# input_folder = '/content/drive/MyDrive/golbahar'
# output_folder = '/content/drive/MyDrive/golbahar_combined'
# os.makedirs(output_folder, exist_ok=True)

# # اندازه برش و اورلپ
# crop_size = 1024
# overlap = 128

# # تابع پردازش برش
# def process_crop(crop):
#     tensor = resize(to_tensor(crop)).unsqueeze(0).to(args.device)
#     with torch.no_grad():
#         model.set_input(tensor, tensor)
#         model.forward()
#         out = model.fake_B.squeeze().cpu().numpy()
#         out = np.clip(out, -1, 1)
#         out = (out + 1) / 2
#         if out.ndim == 2:
#             out = np.stack([out]*3, axis=-1)
#         return out

# # پردازش تمام تصاویر موجود در پوشه ورودی
# for filename in sorted(os.listdir(input_folder)):
#     if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#         continue

#     image_path = os.path.join(input_folder, filename)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"❌ تصویر یافت نشد: {filename}")
#         continue

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_pil = Image.fromarray(image)

#     # برش چپ و راست بدون هیچ شرطی
#     left_crop = image_pil.crop((0, 0, crop_size, crop_size))
#     right_crop = image_pil.crop((image_pil.width - crop_size, 0, image_pil.width, crop_size))

#     # پردازش هر برش
#     left_output = process_crop(left_crop)
#     right_output = process_crop(right_crop)

#     # ترکیب با overlap
#     combined = np.zeros((crop_size, crop_size * 2 - overlap, 3), dtype=np.float32)
#     combined[:, :crop_size] = left_output
#     combined[:, -crop_size:] = right_output

#     for i in range(overlap):
#         alpha = i / (overlap - 1)
#         combined[:, crop_size - overlap + i] = (
#             left_output[:, crop_size - overlap + i] * (1 - alpha) +
#             right_output[:, i] * alpha
#         )

#     # ذخیره خروجی
#     final_image = Image.fromarray((combined * 255).astype(np.uint8))
#     save_path = os.path.join(output_folder, filename)
#     final_image.save(save_path)
#     print(f"✅ ذخیره شد: {save_path}")
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

# # تنظیمات
# args = parser.parse_arguments()
# args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# commons.make_deterministic(args.seed)

# model = network.pix2pix(args, 3, 1)
# if args.resume is not None:
#     print("loading model ....")
#     model = util.resume_model_pix2pix(args, model)
# model.setup()
# model.netG = model.netG.eval()

# to_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=0.5, std=0.5),
# ])
# resize = transforms.Resize(args.GAN_resize)

# input_folder = '/content/drive/MyDrive/golbahar'
# output_folder = '/content/drive/MyDrive/golbahar_combined_512'
# os.makedirs(output_folder, exist_ok=True)

# patch_size = 512
# overlap = 35
# stride = patch_size - overlap

# def process_crop(crop):
#     tensor = resize(to_tensor(crop)).unsqueeze(0).to(args.device)
#     with torch.no_grad():
#         model.set_input(tensor, tensor)
#         model.forward()
#         out = model.fake_B.squeeze().cpu().numpy()
#         out = np.clip(out, -1, 1)
#         out = (out + 1) / 2
#         if out.ndim == 2:
#             out = np.stack([out]*3, axis=-1)
#         return out

# def blend_patches(patches, H, W):
#     result = np.zeros((H, W, 3), dtype=np.float32)
#     weight = np.zeros((H, W, 1), dtype=np.float32)

#     for (x, y, patch) in patches:
#         h, w = patch.shape[:2]
#         patch_weight = np.ones((h, w, 1), dtype=np.float32)

#         # Create a cosine window for soft blending
#         win_x = np.hanning(w)
#         win_y = np.hanning(h)
#         window = np.outer(win_y, win_x)
#         window = window[:, :, None]
#         patch_weight *= window

#         result[y:y+h, x:x+w] += patch * patch_weight
#         weight[y:y+h, x:x+w] += patch_weight

#     result /= np.maximum(weight, 1e-6)
#     return result

# for filename in sorted(os.listdir(input_folder)):
#     if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#         continue

#     image_path = os.path.join(input_folder, filename)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"❌ تصویر یافت نشد: {filename}")
#         continue

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     H, W = image.shape[:2]
#     patches = []

#     y_positions = list(range(0, H - patch_size + 1, stride))
#     x_positions = list(range(0, W - patch_size + 1, stride))

#     # اضافه کردن آخرین ردیف اگر ناقص باقی مانده
#     if y_positions[-1] + patch_size < H:
#         y_positions.append(H - patch_size)

#     if x_positions[-1] + patch_size < W:
#         x_positions.append(W - patch_size)

#     for y in y_positions:
#         for x in x_positions:
#             crop = Image.fromarray(image[y:y+patch_size, x:x+patch_size])
#             output = process_crop(crop)
#             patches.append((x, y, output))

#     final_output = blend_patches(patches, H, W)
#     final_image = Image.fromarray((final_output * 255).astype(np.uint8))
#     save_path = os.path.join(output_folder, filename)
#     final_image.save(save_path)
#     print(f"✅ ذخیره شد: {save_path}")

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
import parser
import time


folder_name = "crops_1"
# تنظیمات پایه
input_dir = r"D:\RPL\Tiles\Dehat\to_thermal"
generated_save_dir = r"D:\RPL\Tiles\Dehat\thermal"
os.makedirs(generated_save_dir, exist_ok=True)

# تنظیمات مدل
args = parser.parse_arguments()
commons.make_deterministic(args.seed)

model = network.pix2pix(args, 3, 1)
if args.resume is not None:
    print("loading model ....")
    model = util.resume_model_pix2pix(args, model)
model.setup()
model.netG = model.netG.eval()

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

# تنظیمات برش
# بارگذاری و پردازش تصاویر
times = []
times2 = []
for i, filename in enumerate(os.listdir(input_dir)):
    if filename.endswith(('.png', '.jpg', '.jpeg','tif')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor((image), cv2.COLOR_BGR2RGB)

        
        image_tensor = resized_transform(Image.fromarray(image))
        image_tensor = image_tensor.unsqueeze(0).to(args.device)
        
        start_time = time.time()

        with torch.no_grad():
            model.set_input(image_tensor, image_tensor)
            model.forward()

            output = model.fake_B
            output = torch.clamp(output, min=-1, max=1)  
            output_images = output * 0.5 + 0.5
            end_time = time.time()
            elapsed = end_time - start_time
            times2.append(elapsed)
            generated_image = transforms.ToPILImage()(output_images.squeeze().cpu())

            generated_image_path = os.path.join(generated_save_dir, filename)
            generated_image.save(generated_image_path)
            
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"✅ Cropped and processed image {i} saved: {generated_image_path}")
        
        

# محاسبه میانگین
if times:
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average processing time per image: {avg_time:.4f} sec")
    
if times2:
    avg_time = sum(times2) / len(times2)
    print(f"\n📊 Average processing time per image2: {avg_time:.4f} sec")