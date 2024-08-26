import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

def augment_images(input_folder, output_folder, num_augments=6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the Keras ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        brightness_range=[0.2,1.0],
        width_shift_range=0.1,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        img = load_img(image_path)  # Load image as PIL object
        x = img_to_array(img)  # Convert image to numpy array
        x = x.reshape((1,) + x.shape)  # Reshape it to (1, width, height, channels)
        
        # Generate multiple augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix=os.path.splitext(image_name)[0], save_format='jpg'):
            i += 1
            if i >= num_augments:
                break  # Stop after generating 'num_augments' images

# Example usage
input_folder = r'C:\Users\US\Desktop\javacode\images'
output_folder = r'C:\Users\US\Desktop\javacode\augmented_images'
augment_images(input_folder, output_folder, num_augments=10)
