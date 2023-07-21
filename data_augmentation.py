import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the paths to your dataset folders
train_dir = r'brain_tumor_mri_dataset_binary\Train'
train_tumor_folder = r'brain_tumor_mri_dataset_binary\Train\tumor'
train_no_tumor_folder = r'brain_tumor_mri_dataset_binary\Train\no_tumor'
train_tumor_augmented_folder = r'brain_tumor_mri_dataset_binary\Augmented\Train\tumor_augmented'
train_no_tumor_augmented_folder = r'brain_tumor_mri_dataset_binary\Augmented\Train\no_tumor_augmented'

test_dir = r'brain_tumor_mri_dataset_binary\Test'
test_tumor_folder = r'brain_tumor_mri_dataset_binary\Test\tumor'
test_no_tumor_folder = r'brain_tumor_mri_dataset_binary\Test\no_tumor'
test_tumor_augmented_folder = r'brain_tumor_mri_dataset_binary\Augmented\Test\tumor_augmented'
test_no_tumor_augmented_folder = r'brain_tumor_mri_dataset_binary\Augmented\Test\no_tumor_augmented'

# Define the desired number of augmented images
desired_train_count = 5000
desired_test_count = 1000

# Functions to apply data augmentation on an image

## flip the image horizontally
def flip_augmentation(image):
    img = image.copy()
    flipped_image = cv2.flip(img, 1)
    return flipped_image

# ## rotate the image 10 degrees clockwise
# def rotate_augmentation(image):
#     img = image.copy()
#     M = cv2.getRotationMatrix2D((img.shape[0]/2,image.shape[1]/2),-10,1) 
#     rotated_image = cv2.warpAffine(image,M,(img.shape[1],img.shape[0])) 
#     return rotated_image

## crop 95%
def crop_augmentation(image):
    height, width = image.shape[:2]
    crop_height = int(height * 0.95)
    crop_width = int(width * 0.95)
    start_x = int((width - crop_width) / 2)
    start_y = int((height - crop_height) / 2)
    cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    return cropped_image

# Function to perform data augmentation on the folder
def save_augmented_folder(folder, output_folder, desired_count):
    folder_images = os.listdir(folder)
    image_count = len(folder_images)
    print(f"The number of images in the folder at: {folder} is {image_count}\n")
    augmentation_factor = desired_count - image_count
    print(f"the number of images to be added are: {augmentation_factor}\n")

    # Copy the original images to the output folder
    for image_name in tqdm(folder_images, desc='Copying Original Images to Augmented folder'):
        image_path = os.path.join(folder, image_name)
        img = cv2.imread(image_path)
        img1 = img.copy()
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, img1)

    for i in tqdm(range(augmentation_factor), desc=f'Augmenting {folder}'):
        if i < len(folder_images):
            image_name = folder_images[i]  # Get the image name from the folder_images list
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)  # Read the image

            augmented_image = flip_augmentation(image)
            output_path = os.path.join(output_folder, f'flipped_{i}_{image_name}')
            cv2.imwrite(output_path, augmented_image)
        else:
            image_name = folder_images[i%image_count]  # Get the image name from the folder_images list
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)  # Read the image

            augmented_image = crop_augmentation(image)
            output_path = os.path.join(output_folder, f'cropped{i}_{image_name}')
            cv2.imwrite(output_path, augmented_image)

# Perform data augmentation

save_augmented_folder(test_tumor_folder, test_tumor_augmented_folder, desired_test_count)
save_augmented_folder(test_no_tumor_folder, test_no_tumor_augmented_folder, desired_test_count)
save_augmented_folder(train_tumor_folder, train_tumor_augmented_folder, desired_train_count)
save_augmented_folder(train_no_tumor_folder, train_no_tumor_augmented_folder, desired_train_count)