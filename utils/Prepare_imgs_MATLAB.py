from PIL import Image
import os

# Get paths of all images from a certain folder (for simpler processing images)
def get_image_paths(folder_path):
    image_paths = []
    for file_name in os.listdir(folder_path):
        image_paths.append(os.path.abspath(os.path.join(folder_path, file_name)))

    print(f'Found {len(image_paths)} images in {folder_path}')
    return image_paths

# Convert to Matlab's acceptable mode
def convert_cmyk_to_rgb_in_folder(folder_path):
    image_paths = get_image_paths(folder_path=folder_path)
    for image_path in image_paths:
            image = Image.open(image_path)
            if image.mode == 'CMYK':
                image = image.convert('RGB')    
                image.save(image_path) # replace old image with rgb image at the same path

# Checking if all images' sizes are the same
def check_imgs_size(folder_path):
    image_paths = get_image_paths(folder_path=folder_path)
    img_sizes = []
    for image_path in image_paths:
            image = Image.open(image_path)
            img_sizes.append(image.size)
    return list(set(img_sizes))

# If they're not the same - reshape
def resize_images(folder_path, dim1, dim2):
    image_paths = get_image_paths(folder_path=folder_path)
    for image_path in image_paths:
            image = Image.open(image_path)
            image = image.resize((dim1, dim2), Image.Resampling.LANCZOS)
            image.save(image_path) 


paths = [] # add full paths to classes folders
for path in paths:
     convert_cmyk_to_rgb_in_folder(path)
     img_sizes = check_imgs_size(path)
     if len(img_sizes) != 1: # more than one unique size
          resize_images(path, 64, 64)