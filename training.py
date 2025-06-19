from utils import parse_label_file
from loss import yolo_loss
from model import C, S, B, IMG_SIZE, yolov1, tiny_yolov1
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

def load_image_and_label(image_path, label_path, num_classes = C):
    def _load_fn(img_path, lbl_path):
        # Load and preprocess image
        img = cv2.imread(img_path.decode(), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # shape: (448, 448, 1)

        # Load label
        label = parse_label_file(lbl_path, S, num_classes)
        return img, label

    image, label = tf.numpy_function(
        func=_load_fn,
        inp=[image_path, label_path],
        Tout=[tf.float32, tf.float32]
    )
    image.set_shape((IMG_SIZE, IMG_SIZE, 1))
    label.set_shape((S, S, num_classes + 5))
    return image, label


def create_yolo_dataset(image_dir, label_dir, num_classes=2, batch_size=16, shuffle=True):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    label_paths = [os.path.join(label_dir, os.path.basename(f).replace('.jpg', '.txt')) for f in image_paths]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    dataset = dataset.map(lambda x, y: load_image_and_label(x, y, num_classes),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)
    
    # Enable memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    train_ds = create_yolo_dataset(
    image_dir='data/images/train',
    label_dir='data/labels/train',
    batch_size=8
    )

    val_ds = create_yolo_dataset(
        image_dir='data/images/val',
        label_dir='data/labels/val',
        batch_size=8,
        shuffle=False
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tensorboard_tiny = tf.keras.callbacks.TensorBoard(log_dir="logs/tiny_yolo", histogram_freq=1)
    checkpoint_tiny = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/tiny_yolo_best.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/yolo", histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/yolo_best.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    os.makedirs("history", exist_ok=True)

    model_tiny = tiny_yolov1()
    learning_rate_tiny = 1e-5
    model_tiny.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_tiny), loss=yolo_loss)
    model_tiny.summary()
    history_tiny = model_tiny.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping, tensorboard_tiny, checkpoint_tiny]
    )
    history_tiny_df = pd.DataFrame(history_tiny.history)
    history_tiny_df.to_csv('history/training_tiny.csv', index=False)

    model = yolov1()
    learning_rate = 1e-5
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=yolo_loss)
    model.summary()
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping, tensorboard, checkpoint]
    )
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('history/training.csv', index=False)