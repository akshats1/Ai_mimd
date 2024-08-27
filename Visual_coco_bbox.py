import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def display_images_with_coco_annotations(image_paths, annotations, display_type='bbox', colors=None, label="Positive"):
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
        
        # Display bounding box and label
        if display_type == 'bbox':
            bbox = ann['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=color, facecolor='none')
            axs.add_patch(rect)
            
            # Add label text above the bounding box
            axs.text(bbox[0], bbox[1] - 10, label, color=color, fontsize=12, backgroundcolor='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Load COCO annotations
with open('Image_Aks/train.json', 'r') as f:
    annotations = json.load(f)

# Get all image files
image_dir = "Image_Aks/"
all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
random_image_files = random.sample(all_image_files, 1)

# Display only the bounding box with label "Positive"
display_type = 'bbox'
display_images_with_coco_annotations(random_image_files, annotations, display_type)

