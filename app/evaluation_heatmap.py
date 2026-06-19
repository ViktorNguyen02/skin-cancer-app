import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model

# === Load trained model ===
model = load_model("skin_cancer_model_25.h5")
print("✅ Model loaded.")

# === Pick an image ===
img_path = "/Users/couoquevietnoullaine/Documents/cancer/data/HAM10000_images_part_2/ISIC_0029308.jpg"  # or part_1

assert os.path.exists(img_path), f"Image not found: {img_path}"

# === Load and preprocess image ===
IMG_SIZE = 128

def load_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

img_array, original_img = load_image(img_path)

# === Grad-CAM ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

heatmap = make_gradcam_heatmap(img_array, model, "conv2d_2")

def overlay_heatmap(heatmap, img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(img)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

overlay_img = overlay_heatmap(heatmap, original_img)

plt.figure(figsize=(6, 6))
plt.imshow(overlay_img)
plt.title("Grad-CAM for Selected Image")
plt.axis('off')
plt.show()
