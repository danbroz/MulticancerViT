import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
import itertools
import tensorflow as tf
from matplotlib import gridspec
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report

# Hyperparameters
class Hyperparameters:
    def __init__(self):
        self.image_size = 512
        self.num_channels = 3
        self.patch_size = 64
        self.num_patches = (self.image_size**2) // (self.patch_size**2)
        self.flat_patches_shape = (self.num_patches, self.patch_size*self.patch_size*self.num_channels)
        self.batch_size = 32
        self.lr = 1e-4
        self.num_epochs = 30
        self.num_classes = 22
        self.num_layers = 12
        self.hidden_dim = 512
        self.mlp_dim = 3072
        self.num_heads = 12
        self.dropout_rate = 0.1
        self.class_names = None

# Helper functions
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_class_name(root_dir):
    classes_name = []
    r_folder = os.listdir(root_dir)
    for i in r_folder:
        if i != "ALL":
            list_classes = os.listdir(os.path.join(root_dir, i))
            for j in list_classes:
                classes_name.append(j)
    return classes_name

def get_image_paths(root_dir, split=0.09):
    image_paths = []
    cancer_fol = os.listdir(root_dir)
    for cancer_name in cancer_fol:
        if cancer_name != "ALL":
            class_names = os.listdir(os.path.join(root_dir, cancer_name))
            for class_name in class_names:
                class_path = os.path.join(root_dir, cancer_name, class_name)
                images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
                image_paths.extend(images)
    
    split_rate = int(len(image_paths) * split)
    train, valid = train_test_split(image_paths, test_size=split_rate, random_state=42)
    train, test = train_test_split(train, test_size=split_rate, random_state=42)
    return train, valid, test

def process_image_label(path, hp):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp.image_size, hp.image_size))
    image = image / 255.0
    
    patch_shape = (hp.patch_size, hp.patch_size, hp.num_channels)
    patches = patchify(image, patch_shape, hp.patch_size)
    
    patches = np.reshape(patches, hp.flat_patches_shape)
    patches = patches.astype(np.float32)
    
    # class name
    class_name = path.split("/")[-2]
    class_idx = hp.class_names.index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)
    
    return patches, class_idx

def parse(path, hp):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp.num_classes)
    
    patches.set_shape(hp.flat_patches_shape)
    labels.set_shape(hp.num_classes)
    
    return patches, labels

def tf_dataset(images, hp):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(lambda x: parse(x, hp))
    ds = ds.batch(hp.batch_size).prefetch(8)
    return ds

# Model architecture
def create_vit_model(hp):
    # Input layer
    inputs = layers.Input(shape=hp.flat_patches_shape)
    
    # Patch embeddings
    x = layers.Dense(hp.hidden_dim)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Transformer blocks
    for _ in range(hp.num_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=hp.num_heads,
            key_dim=hp.hidden_dim // hp.num_heads
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # MLP
        mlp_output = layers.Dense(hp.mlp_dim, activation='gelu')(x)
        mlp_output = layers.Dropout(hp.dropout_rate)(mlp_output)
        mlp_output = layers.Dense(hp.hidden_dim)(mlp_output)
        x = layers.Add()([x, mlp_output])
        x = layers.LayerNormalization()(x)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp.dropout_rate)(x)
    outputs = layers.Dense(hp.num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Initialize hyperparameters
    hp = Hyperparameters()
    
    # Set paths
    root_path = "/kaggle/input/multi-cancer/Multi Cancer/"
    model_path = "/kaggle/working/MultiCancerViT.h5"
    csv_path = "/kaggle/working/MultiCancerViT.csv"
    
    # Get class names
    hp.class_names = get_class_name(root_path)
    
    # Get image paths
    train, valid, test = get_image_paths(root_path)
    print(f"Train:{len(train)} - Valid: {len(valid)} - Test:{len(test)}")
    
    # Create datasets
    train_ds = tf_dataset(train, hp)
    valid_ds = tf_dataset(valid, hp)
    test_ds = tf_dataset(test, hp)
    
    # Create and compile model
    model = create_vit_model(hp)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.CSVLogger(csv_path),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=hp.num_epochs,
        callbacks=callbacks_list
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main() 