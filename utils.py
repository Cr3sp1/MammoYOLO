from model import C, S, B, IMG_SIZE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import os
import cv2

# Return a (grid_size, grid_size, num_classes + 5) size label tensor 
def parse_label_file(filepath, grid_size=S, num_classes=C):
    y_true = np.zeros((grid_size, grid_size, num_classes + 5), dtype=np.float32)
    if not os.path.isfile(filepath):
            print(f"Label file {filepath} not found!")
    with open(filepath, 'r') as f:
        for line in f:
            class_id, x_center, y_center, bw, bh = map(float, line.strip().split())

            # find grid cell
            cell_x = int(x_center * grid_size)
            cell_y = int(y_center * grid_size)

            # relative position within the cell
            rel_x = x_center * grid_size - cell_x
            rel_y = y_center * grid_size - cell_y

            # Set one-hot class
            y_true[cell_y, cell_x, :num_classes] = tf.keras.utils.to_categorical(int(class_id), num_classes)

            # Set box coordinates and confidence
            y_true[cell_y, cell_x, num_classes:num_classes+4] = [rel_x, rel_y, bw, bh]
            y_true[cell_y, cell_x, num_classes+4] = 1.0  # response_mask

            # We only handle one object per cell (like original YOLOv1)
    return y_true


def label_tensor_to_boxes(label_tensor, grid_size=S, num_classes=C):
    """
    Tranform ground truth label tensor to a np.array with shape (n_boxes, 5)
    containing all boxes with format [class_id, x_center, y_center, bw, bh].
    """
    boxes = []
 
    for i in range(grid_size):
        for j in range(grid_size):
            cell = label_tensor[i, j]

            response = cell[num_classes + 4]
            if response == 1:
                class_id = np.argmax(cell[:num_classes])
                cx, cy, bw, bh = cell[num_classes:num_classes+4]

                x_center = (j + cx)/ grid_size
                y_center = (i + cy)/ grid_size

                boxes.append([class_id, x_center, y_center, bw, bh])
    
    return np.array(boxes)

def pred_tensor_to_boxes(label_tensor, grid_size=S, num_classes=C, num_boxes=B):
    boxes = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell = label_tensor[i, j]
            cond_class_probs = cell[:num_classes]

            for b in range(num_boxes):
                base = num_classes + b * 5
                cx, cy, bw, bh, conf = cell[base : base + 5]

                # Convert to global image coordinates (normalized)
                x_center = (j + cx) / grid_size
                y_center = (i + cy) / grid_size

                # Confidence-weighted class probabilities
                class_scores = cond_class_probs * conf

                # For multi-class: store all class probabilities
                boxes.append(np.concatenate([class_scores, [x_center, y_center, bw, bh, conf]]))

    return np.array(boxes)



def draw_yolo_labels(image, label_tensor, num_classes=C, grid_size=S, class_names=["benign", "malignant"]):
    """
    Draws YOLO-style bounding boxes on the image based on label tensor.
    `label_tensor` is (grid_size, grid_size, num_classes + 5)
    """
    h, w = image.shape[:2]
    cell_w, cell_h = w / grid_size, h / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            cell = label_tensor[i, j]
            response = cell[num_classes + 4]
            if response == 1:
                class_id = np.argmax(cell[:num_classes])
                cx, cy, bw, bh = cell[num_classes:num_classes+4]

                # Absolute position
                abs_cx = (j + cx) * cell_w
                abs_cy = (i + cy) * cell_h
                abs_w = bw * w
                abs_h = bh * h

                x1 = int(abs_cx - abs_w / 2)
                y1 = int(abs_cy - abs_h / 2)
                x2 = int(abs_cx + abs_w / 2)
                y2 = int(abs_cy + abs_h / 2)

                label = class_names[class_id] if class_names else f"Class {class_id}"
                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def show_examples(image_dir, label_dir, num_classes=C, grid_size=S, class_names=["benign", "malignant"], num_examples=5):
    """
    Displays a few labeled training examples.
    """
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    for img_file, label_file in zip(image_files[:num_examples], label_files[:num_examples]):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.isfile(img_path):
            print(f"Image file {img_path} not found!")
            return
        if not os.path.isfile(label_path):
            print(f"Label file {label_path} not found!")
            return

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        label_tensor = parse_label_file(label_path, grid_size=grid_size, num_classes=num_classes)

        vis_img = draw_yolo_labels(image.copy(), label_tensor, num_classes=num_classes, grid_size=grid_size, class_names=class_names)

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(img_file)
        plt.axis('off')
        plt.show()


def show_batch(img_ids, img_dir="data/images/train", label_dir="data/labels/train", num_classes=C, grid_size=S, cols=3, class_names=["benign", "malignant"]):
    rows = (len(img_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img_id in enumerate(img_ids):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        img_path = os.path.join(img_dir, img_id + '.jpg')
        label_path = os.path.join(label_dir, img_id + '.txt')

        if not os.path.isfile(img_path):
            print(f"Image file {img_path} not found!")
            continue
        if not os.path.isfile(label_path):
            print(f"Label file {label_path} not found!")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (448, 448))

        label = parse_label_file(label_path, grid_size=grid_size, num_classes=num_classes)

        ax.imshow(img, cmap='gray')
        ax.set_title(img_id)

        for row in range(grid_size):
            for col in range(grid_size):
                cell = label[row, col]
                response = cell[num_classes + 4]
                if response > 0:
                    bbox = cell[num_classes:num_classes+4]
                    class_id = np.argmax(cell[:num_classes])
                    label_text = class_names[class_id] if class_names else f"Class {class_id}"

                    x_center = (col + bbox[0]) / grid_size * 448
                    y_center = (row + bbox[1]) / grid_size * 448
                    bw = bbox[2] * 448
                    bh = bbox[3] * 448
                    x_min = x_center - bw / 2
                    y_min = y_center - bh / 2

                    rect = patches.Rectangle((x_min, y_min), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x_min, y_min - 5, label_text, color='red', fontsize=8, backgroundcolor='white')

        ax.axis('off')

    # Hide unused axes if any
    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()

    