import cv2
import os
import matplotlib.pyplot as plt

# # Directories
# image_dir = r"C:\Users\Acer\Desktop\augmented\testing\augmented_images"  # Augmented images
# label_dir = r"C:\Users\Acer\Desktop\augmented\testing\augmented_labels"  # Corresponding annotations



# Directories
image_dir = r"C:\Users\Bisoj Pc\Desktop\WasteNot\augmented\images"  # Augmented images
label_dir = r"C:\Users\Bisoj Pc\Desktop\WasteNot\augmented\labels"  # Corresponding annotations

# List all images
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Define a function to visualize YOLO annotations
def visualize_yolo_annotation(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Read annotation file
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id, x_center, y_center, w, h = map(float, data)

        # Convert YOLO format to OpenCV format (absolute coordinates)
        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)

        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"Class {class_id}", (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show image with bounding boxes
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
# Test visualization on a few images
for i in range(100):  # Display 5 sample images
    image_path = os.path.join(image_dir, image_files[i])
    label_path = os.path.join(label_dir, image_files[i].replace(".jpg", ".txt"))
    
    print(f"Displaying: {image_files[i]}")
    visualize_yolo_annotation(image_path, label_path)
    

