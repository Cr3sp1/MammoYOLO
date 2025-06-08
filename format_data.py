import os
import pydicom
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

# --- CONFIG --- #
manifest_dir = "data/manifest-ZkhPvrLo5216730872708713142"
root_dir = os.path.join(manifest_dir, "CBIS-DDSM")
output_dir = "data"
image_out_dir = os.path.join(output_dir, "images")
label_out_dir = os.path.join(output_dir, "labels")
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(image_out_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_out_dir, split), exist_ok=True)

lesion_files = {
    "data/mass_case_description_train_set.csv": "train",
    "data/mass_case_description_test_set.csv": "test",
    "data/calc_case_description_train_set.csv": "train",
    "data/calc_case_description_test_set.csv": "test"
}

# --- LOAD METADATA --- #
metadata = pd.read_csv(os.path.join(manifest_dir, "metadata.csv"))
metadata.columns = metadata.columns.str.strip()
roi_meta = metadata[metadata['Series Description'] == 'ROI mask images']
full_meta = metadata[metadata['Series Description'] == 'full mammogram images']

# --- UTILS --- #
def pathology_to_class(p): return 1 if p == "MALIGNANT" else 0

def load_dicom_array(path):
    arr = pydicom.dcmread(path).pixel_array
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)

def get_bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None

# --- BUILD LESION MAP --- #
lesion_map = defaultdict(list)
for lesion_csv, split in lesion_files.items():
    print(f"Processing {lesion_csv}...")
    if not os.path.exists(lesion_csv):
        print(f"⚠️ Missing: {lesion_csv}")
        continue

    lesions = pd.read_csv(lesion_csv)
    # print(f"Number of lesions found: {len(lesions)}")
    for _, row in tqdm(lesions.iterrows(), total=len(lesions), desc=f"Indexing {split}"):
        roi_dir = os.path.dirname(row['ROI mask file path'].replace("\\", "/").strip())
        roi_path = os.path.join(root_dir, roi_dir, "1-2.dcm")
        if not os.path.isfile(roi_path): 
            roi_path = os.path.join(root_dir, roi_dir, "1-1.dcm")
        full_dir = os.path.dirname(row['image file path'].replace("\\", "/").strip())
        full_path = os.path.normpath(os.path.join(root_dir, full_dir, "1-1.dcm"))
        pathology = pathology_to_class(row['pathology'].strip().upper())

        if os.path.isfile(full_path) and os.path.isfile(roi_path):
            lesion_map[(full_path, split)].append((roi_path, pathology))
            # print ('directory found')
            continue
        if not os.path.isfile(full_path):
            print(f"Full image {full_path} not found!")
        if not os.path.isfile(roi_path):
            print(f"Roi image {roi_path} not found!")

# --- PROCESS ALL FULL MAMMOGRAMS (POSITIVE + NEGATIVE) --- #
train_entries = []
all_full_splits = {
    os.path.normpath(os.path.join(manifest_dir, row['File Location'], "1-1.dcm")):
    'train' if 'train' in row['File Location'].lower() else 'test'
    for _, row in full_meta.iterrows()
}

target_size = 448       # We want 448X448 images
for full_path, split in tqdm(all_full_splits.items(), desc="Processing all full mammograms"):
    if not os.path.isfile(full_path):
        print(f"⚠️ Full image missing: {full_path}")
        continue

    try:
        image = load_dicom_array(full_path)
        h, w = image.shape
        scale_x = target_size / w
        scale_y = target_size / h
        resized_image = cv2.resize(image, (target_size, target_size))
        img_name = full_path.split(os.sep)[3]
        img_path = os.path.join(image_out_dir, split, img_name + ".jpg")
        # print(f"Processing full image {img_name}")
        label_path = os.path.join(label_out_dir, split, img_name + ".txt")

        labels = []
        for roi_path, class_id in lesion_map.get((full_path, split), []):
            if not os.path.isfile(roi_path):
                continue
            mask = load_dicom_array(roi_path)
            if mask.shape != image.shape:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y, bw, bh = bbox
                x = int(x * scale_x)
                y = int(y * scale_y)
                bw = int(bw * scale_x)
                bh = int(bh * scale_y)
                x_center = (x + bw / 2) / target_size
                y_center = (y + bh / 2) / target_size
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw/target_size:.6f} {bh/target_size:.6f}")

        cv2.imwrite(img_path, resized_image)
        with open(label_path, 'w') as f:
            f.write("\n".join(labels) + "\n" if labels else "")

        if split == "train":
            train_entries.append(img_path)
    except Exception as e:
        print(f"⚠️ Error processing {full_path}: {e}")

# --- SPLIT TRAIN/VAL --- #
if train_entries:
    print("\n🔀 Splitting train into train/val...")
    train_imgs, val_imgs = train_test_split(train_entries, test_size=0.2, random_state=42)
    for path in val_imgs:
        fname = os.path.basename(path)
        os.rename(path, os.path.join(image_out_dir, "val", fname))
        os.rename(os.path.join(label_out_dir, "train", fname.replace(".jpg", ".txt")),
                  os.path.join(label_out_dir, "val", fname.replace(".jpg", ".txt")))

print("\n✅ Done. Check: data/images/ and data/labels/")