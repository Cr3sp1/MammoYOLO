import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from model import IMG_SIZE, C, S
from sklearn.metrics import roc_curve, roc_auc_score
from training import create_yolo_dataset
from model import yolov1, tiny_yolov1
from utils import label_tensor_to_boxes, pred_tensor_to_boxes


# ---- Utility functions ----
def center_inside_box(center, box):
    if not isinstance(box, (list, np.ndarray)) or len(box) != 4:
        print(f"Invalid box: {box}")
        return False

    bx, by, bw, bh = box 
    x, y = center

    return (bx - bw / 2) <= x <= (bx + bw / 2) and (by - bh / 2) <= y <= (by + bh / 2)

def collect_predictions(model, dataset):
    all_pred_boxes = []
    all_gt_boxes = []
    y_true = []
    y_scores = []
    n_images = 0

    for images, labels in dataset:
        preds = model(images, training=False).numpy()
        labels = labels.numpy()
        batch_size = images.shape[0]

        for i in range(batch_size):
            pred = preds[i]
            pred_boxes = pred_tensor_to_boxes(pred)
            all_pred_boxes.append(pred_boxes)

            label_tensor = labels[i]
            gt_boxes = label_tensor_to_boxes(label_tensor)

            # Get GT boxes for class 1 (malignant)
            gt_boxes = gt_boxes[gt_boxes[:, 0] == 1][:, 1:]
            all_gt_boxes.append(gt_boxes)
            y_true.append(1 if len(gt_boxes) != 0 else 0)
            n_images += 1

            max_score = 0
            for row in pred_boxes:
                # Assuming row[1] corresponds to malignant class score
                score = row[1]
                max_score = max(max_score, score)
            
            y_scores.append(max_score)
                

    return all_pred_boxes, all_gt_boxes, n_images, np.array(y_true), np.array(y_scores)

def compute_froc_from_predictions(pred_boxes_lists, gt_boxes_list, n_images, thresholds=np.linspace(0.0, 1.0, 50)):
    froc_points = []
    total_gt = sum(len(gt) for gt in gt_boxes_list)

    for thresh in thresholds:
        tp = 0
        fp = 0

        for pred_boxes, gt_boxes in zip(pred_boxes_lists, gt_boxes_list):
            pred_filtered = [p for p in pred_boxes if p[1] >= thresh]
            pred_filtered.sort(key=lambda x: -x[1])
            matched = np.zeros(len(gt_boxes), dtype=bool)

            for pred in pred_filtered:
                center = pred[2:4]
                found = False
                for i, gt_box in enumerate(gt_boxes):
                    if center_inside_box(center, gt_box) and not matched[i]:
                        matched[i] = True
                        tp += 1
                        found = True
                        break
                if not found:
                    fp += 1

        sensitivity = tp / total_gt if total_gt else 0
        fp_per_image = fp / n_images if n_images else 0
        froc_points.append((fp_per_image, sensitivity, thresh))

    return np.array(froc_points)


# ---- Main Evaluation ----

def evaluate_and_save(model, test_dataset, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    print("Running model on test set...")
    preds, gts, n_imgs, y_true, y_scores = collect_predictions(model, test_dataset)

    print("Computing ROC and AUC...")
    fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)
    roc_thresh[0] = 1
    roc_auc = roc_auc_score(y_true, y_scores)

    roc_df = pd.DataFrame({
        "FalsePositiveRate": fpr,
        "TruePositiveRate": tpr,
        "Threshold": roc_thresh,
        "ROC_AUC": [roc_auc] * len(fpr)  # Repeat AUC for each row
    })

    roc_df.to_csv(os.path.join(output_dir, model_name + "_roc_curve.csv"), index=False)

    print("Computing FROC...")
    froc_points = compute_froc_from_predictions(preds, gts, n_imgs)
    froc_df = pd.DataFrame(froc_points, columns=["FalsePositivesPerImage", "Sensitivity", "Threshold"])
    froc_df.to_csv(os.path.join(output_dir, model_name + "_froc_curve.csv"), index=False)

    print(f"Saved ROC and FROC curves to '{output_dir}'")


# ---- Script ----

if __name__ == "__main__":
    # Load dataset and models
    test_dataset = create_yolo_dataset('data/images/test', 'data/labels/test', batch_size=1)
    model_tiny = tiny_yolov1()
    model = yolov1()
    model_tiny.load_weights('checkpoints/tiny_yolo_best.h5')
    model.load_weights('checkpoints/yolo_best.h5')

    
    output_dir = "performance"
    # Evaluate and save
    evaluate_and_save(model_tiny, test_dataset, output_dir, "tiny_yolo")
    evaluate_and_save(model, test_dataset, output_dir, "yolo")