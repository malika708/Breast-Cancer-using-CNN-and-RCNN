import os
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

def generate_dataset_from_images(data_dir, img_size=64):
    """Generate dataset from raw images in the specified directory."""
    images = []
    labels = []
    label_dict = {'Type-D': 0, 'Type-L': 1, 'Type-M': 2}
    
    print(f"Searching for images in: {data_dir}")
    print(f"Looking for cancer types: {list(label_dict.keys())}")
    
    # Check if we need to go one level deeper
    inner_dir = os.path.join(data_dir, "BreaKHis_v1")
    if os.path.exists(inner_dir):
        data_dir = inner_dir
        print(f"Found inner BreaKHis_v1 directory: {data_dir}")
    
    total_images = 0
    for label in label_dict.keys():
        # Search for images in the type directory
        type_dir = os.path.join(data_dir, label)
        if not os.path.exists(type_dir):
            print(f"Warning: Directory not found: {type_dir}")
            continue
            
        # Search for images recursively
        image_paths = glob.glob(os.path.join(type_dir, "**", "*.png"), recursive=True)
        print(f"Found {len(image_paths)} images in {label}")
        
        for path in image_paths:
            try:
                img = load_img(path, target_size=(img_size, img_size))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label_dict[label])
                total_images += 1
                if total_images % 100 == 0:  # Progress update every 100 images
                    print(f"Loaded {total_images} images so far...")
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                continue
    
    if total_images == 0:
        print("No images were loaded. Please check the directory structure.")
        print("Expected structure: BreaKHis_v1/Type-D/patient_id/image.png")
        print(f"Current directory contents: {os.listdir(data_dir)}")
        return None, None
    
    print(f"Successfully loaded {total_images} images")
    
    X = np.array(images).astype('float32') / 255.0
    Y = to_categorical(np.array(labels), num_classes=3)
    
    # Save the processed data
    os.makedirs('model', exist_ok=True)
    np.save('model/X.txt.npy', X)
    np.save('model/Y.txt.npy', Y)
    
    return X, Y

def load_or_generate_dataset(data_dir, img_size=64):
    """Try to load existing dataset, generate if not found."""
    try:
        # Check if the data directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found: {data_dir}")
            return None, None
            
        # Try to load existing .npy files
        if os.path.exists('model/X.txt.npy') and os.path.exists('model/Y.txt.npy'):
            print("Loading existing dataset from .npy files")
            X = np.load('model/X.txt.npy')
            Y = np.load('model/Y.txt.npy')
            return X, Y
        else:
            print("No existing dataset found. Generating from images...")
            return generate_dataset_from_images(data_dir, img_size)
    except Exception as e:
        print(f"Error in load_or_generate_dataset: {str(e)}")
        return None, None

def load_dynamic_dataset(base_dir, classification_type, magnification, img_size=64):
    """
    Dynamically load images and labels from a structure like:
    base_dir/classification_type/magnification/class_name/image.png
    """
    images = []
    labels = []
    class_dir = os.path.join(base_dir, classification_type, magnification)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return None, None, None

    # Dynamically find all class subfolders
    class_names = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
    class_names.sort()  # For consistent label assignment
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"Detected classes: {class_to_idx}")

    total_images = 0
    for class_name in class_names:
        folder = os.path.join(class_dir, class_name)
        image_paths = glob.glob(os.path.join(folder, "*.png"))
        print(f"Found {len(image_paths)} images in class '{class_name}'")
        for path in image_paths:
            try:
                img = load_img(path, target_size=(img_size, img_size))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_to_idx[class_name])
                total_images += 1
                if total_images % 100 == 0:
                    print(f"Loaded {total_images} images so far...")
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                continue
    if total_images == 0:
        print("No images were loaded. Please check the directory structure.")
        print(f"Current directory contents: {os.listdir(class_dir)}")
        return None, None, None
    print(f"Successfully loaded {total_images} images")
    X = np.array(images).astype('float32') / 255.0
    Y = to_categorical(np.array(labels), num_classes=len(class_names))
    return X, Y, class_names 