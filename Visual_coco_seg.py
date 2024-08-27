import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

def display_images_with_coco_annotations(image_paths, annotations, display_type='seg', colors=None, label="Positive"):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    
    # Load the image using OpenCV and convert it from BGR to RGB color space
    image = cv2.imread(image_paths[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    axs.imshow(image)
    axs.axis('off')  # Turn off the axes

    # Define a default color map if none is provided
    if colors is None:
        colors = plt.cm.get_cmap('tab10')

    # Get image filename to match with annotations
    img_filename = os.path.basename(image_paths[0])
    img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']
    
    # Filter annotations for the current image
    img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
    
    for ann in img_annotations:
        category_id = ann['category_id']
        color = colors(category_id % 10)
        
        # Display segmentation mask
        if display_type in ['seg', 'both']:
            for seg in ann['segmentation']:
                # Create a polygon for the segmentation
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=True, facecolor=color, alpha=0.4)
                axs.add_patch(polygon)
    
    # Display the label "Positive" at the bottom of the figure
    fig.text(0.5, 0.01, label, ha='center', va='center', fontsize=14, fontweight='bold', color='black',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))  # Transparent background with alpha

    plt.tight_layout()
    plt.show()

# Load COCO annotations
with open('Image_Aks/train.json', 'r') as f:
    annotations = json.load(f)

# Get all image files
image_dir = "Image_Aks/"
all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
random_image_files = random.sample(all_image_files, 1)

# Display segmentation mask with the label "Positive" at the bottom
display_type = 'seg'
display_images_with_coco_annotations(random_image_files, annotations, display_type)

