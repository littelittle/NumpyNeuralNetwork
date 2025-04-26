import numpy as np
from scipy.ndimage import rotate, shift
from skimage.transform import resize
import time

def rotate_image(image, angle):
    rotate_image = rotate(image, angle, reshape=False, cval=0)
    return rotate_image

def resize_image(image, scale_factor):
    resized_image = resize(image, (int(28 * scale_factor), int(28 * scale_factor)), anti_aliasing=True)
    resized_image_28x28 = resize(resized_image, (28, 28), anti_aliasing=True)
    return resized_image_28x28

def translate_image(image, shift_x, shift_y):
    translation = (shift_y, shift_x)
    translated_image = shift(image, translation, cval=0.0)
    return translated_image

def augment_image(image):
    # Randomly choose an augmentation method
    augmentation_type = np.random.choice(['rotate', 'resize', 'translate'])
    
    if augmentation_type == 'rotate':
        angle = np.random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
        augmented_image = rotate_image(image, angle)
    elif augmentation_type == 'resize':
        scale_factor = np.random.uniform(0.8, 1.2)  # Resize between 80% and 120%
        augmented_image = resize_image(image, scale_factor)
    elif augmentation_type == 'translate':
        shift_x = np.random.randint(-2, 3)  # Shift between -2 and 2 pixels in x direction
        shift_y = np.random.randint(-2, 3)  # Shift between -2 and 2 pixels in y direction
        augmented_image = translate_image(image, shift_x, shift_y)
    
    return augmented_image


if __name__ == "__main__":

    bs = 32
    imgs = np.random.rand(bs, 28*28)
    imgs = imgs.reshape(bs, 28, 28)
    start_time = time.time()
    imgs = np.array([augment_image(img) for img in imgs])
    total_time = time.time() - start_time
    print(f"Time taken for augmentation: {total_time:.4f} seconds")
    print(imgs.shape)