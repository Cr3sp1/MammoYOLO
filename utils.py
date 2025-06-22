from model import C, S, B, IMG_SIZE
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import os
import cv2

# Return a (grid_size, grid_size, num_classes + 5) size label tensor 
def parse_label_file(filepath, grid_size=S, num_classes=C):
    """
    Parse label file into a np tensor with shape (S, S, C + 5), each one of the SxS
    grid cell is associated with a label [one_hot_class, rel_x, rel_y, bw, bh] 
    """
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
    """
    Tranform prediction tensor to a np.array with shape (n_boxes, 5)
    containing all boxes with format [class_scores, x_center, y_center, bw, bh].
    """
    boxes = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell = label_tensor[i, j]
            class_probs = cell[:num_classes]  # shape: (C,)
            confidences = cell[num_classes:num_classes + num_boxes]  # shape: (B,)
            box_data = cell[num_classes + num_boxes:]  # shape: (B*4,)

            for b in range(num_boxes):
                conf = confidences[b]
                box_offset = b * 4
                cx, cy, bw, bh  = box_data[box_offset : box_offset + 4]

                # Convert center coords from cell-relative to global image-relative
                x_center = (j + cx) / grid_size
                y_center = (i + cy) / grid_size

                # Confidence-weighted class scores
                class_scores = class_probs * conf

                boxes.append(np.concatenate([class_scores, [x_center, y_center, bw, bh, conf]]))

    return np.array(boxes)


def show_batch(img_ids, img_dir="data/images/train", label_dir="data/labels/train", num_classes=C, grid_size=S, cols=3, class_names=["Benign", "Malignant"]):
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
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        ax.imshow(img, cmap='gray')
        ax.set_title(img_id)

        label_tensor = parse_label_file(label_path, grid_size=grid_size, num_classes=num_classes)
        label_boxes = label_tensor_to_boxes(label_tensor)
        for box in label_boxes:
            class_id, x_center, y_center, bw, bh = box
            label_text = class_names[int(class_id)] if class_names else f"Class {class_id}"
            bw *= IMG_SIZE
            bh *= IMG_SIZE
            x_min= x_center*IMG_SIZE - bw / 2
            y_min= y_center*IMG_SIZE - bw / 2
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


def show_pred_example(model:Model, image_path, label_path, n_pred=3, title = "", num_classes=C, grid_size=S):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.axis("off")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    label_tensor = parse_label_file(label_path, grid_size=grid_size, num_classes=num_classes)
    label_boxes = label_tensor_to_boxes(label_tensor) 
    for box in label_boxes:
        class_id, x_center, y_center, bw, bh = box
        label_text = "Benign" if class_id == 0 else "Malignant"
        bw *= IMG_SIZE
        bh *= IMG_SIZE
        x_min= x_center*IMG_SIZE - bw / 2
        y_min= y_center*IMG_SIZE - bh / 2
        rect = patches.Rectangle((x_min, y_min), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, label_text, color='red', fontsize=8, backgroundcolor='white')
    
    img_norm = img.astype(np.float32) / 255.0
    img_norm = np.expand_dims(img_norm, axis=-1)  # shape: (448, 448, 1)
    img_batch = np.expand_dims(img_norm, axis=0)  # shape: (1, 448, 448, 1)
    pred_tensor = model(img_batch, training=False).numpy()[0]
    pred_boxes = pred_tensor_to_boxes(pred_tensor)

    sorted_boxes = pred_boxes[pred_boxes[:, num_classes +4].argsort()[::-1]]
    for i in range(0, n_pred):
        box = sorted_boxes[i]
        # print(box)
        class_scores = box[:num_classes]
        x_center, y_center, bw, bh, conf= box[num_classes:]
        
        pred_text = f"Ben score: {class_scores[0]:.3f}\nMal score {class_scores[1]:.3f}"
        bw *= IMG_SIZE
        bh *= IMG_SIZE
        x_min= x_center*IMG_SIZE - bw / 2
        y_min= y_center*IMG_SIZE - bh / 2
        rect = patches.Rectangle((x_min, y_min), bw, bh, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min + bh + 5, pred_text, color='yellow', fontsize=8, backgroundcolor='black')


    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    