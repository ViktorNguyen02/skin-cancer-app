#%%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, BatchNormalization, GlobalAveragePooling2D, Add, Activation, SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

print("✅ All libraries imported successfully!")

print("\n--- HARDWARE CHECK ---")
print(f"TensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"✅ SUCCESS: TensorFlow found the Mac GPU! ({len(gpu_devices)} device(s) available)")
else:
    print("❌ ERROR: TensorFlow is still stuck on the CPU.")
print("----------------------\n")


#%%
# ======================================================
# Load Metadata and Paths
# ======================================================

# Automatically find the folder where this script is located
# __file__ is the current script. abspath gets its full path.
# dirname strips the filename, leaving us with the folder path.
# dirname is programmed with a specific rule: Find the very last slash in the string and throw away everything after it.
# will return "/Users/couoquevietnoullaine/developer/python_projects/skin-cancer-app
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location
# We remove the hardcoded '/Users/couoquevietnoullaine/...'
metadata_path = os.path.join(script_dir, 'data', 'HAM10000_metadata.csv')
images_path_1 = os.path.join(script_dir, 'data', 'HAM10000_images_part_1')
images_path_2 = os.path.join(script_dir, 'data', 'HAM10000_images_part_2')

# Load the CSV with a safety check
if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    print(f"Successfully loaded metadata: {len(df)} rows found.")
else:
    print(f"ERROR: Could not find CSV at {metadata_path}")
    # This helps you debug if the 'data' folder is missing
    raise FileNotFoundError("Check your project structure!")
print(df.info())           
print(df.head())  
#%%
# ==========================================================
# Class Distribution Check
# ==========================================================
print("\n📊 Total count per lesion class:")
print(df['dx'].value_counts())

# Optional: See the percentage breakdown instead of raw counts
print("\n📈 Percentage breakdown per class:")
print(df['dx'].value_counts(normalize=True) * 100)

#%%
# ==========================================================
# THE SEARCH ENGINE (Finding the Images)
# ==========================================================

#In HAM10000, the image and label are "divorced." The CSV has the image ID and label, but not the full path to the image file.

def find_image_path(image_id):
    # Check if the image exists in the 'part_1' folder
    path_1 = os.path.join(images_path_1, image_id + '.jpg')
    path_2 = os.path.join(images_path_2, image_id + '.jpg')
    
    if os.path.exists(path_1):
        return path_1
    elif os.path.exists(path_2):
        return path_2
    else:
        # Raise an explicit error immediately when image is missing
        raise FileNotFoundError(
            f"Image {image_id}.jpg not found in either folder:\n"
            f"  Checked: {path_1}\n"
            f"  Checked: {path_2}"
        )

# Map the paths and encode labels
# 1.to get the full address of every image.
df['path'] = df['image_id'].map(find_image_path)

# 2.Backup the names: Keep a copy of the original disease names (like 'mel') 
# so we can still read them later.
# the correct answers
df['label_str'] = df['dx']

# 3. Setup the Translator: Initialize the tool that turns words into numbers.
label_encoder = LabelEncoder()

# 4. Final Translation: Convert text (e.g., 'nv') into computer-friendly 
# .fit() finds unique classes & maps them alphabetically to numbers (akiec=0, bcc=1... nv=5).
# .transform() swaps the text for those numbers row-by-row. transforms the text into numbers
# df['label'] = creates the new column on the fly and dumps the numbers in.
df['label'] = label_encoder.fit_transform(df['dx'])

# Show the results: Print the first 10 rows to verify the ID, name, 
# number, and file path all match up.
print(df[['image_id', 'dx', 'label', 'path']].head(10))

print("After:", df.columns)

# Get the path for the very first image
sample_path = df['path'][0]

# Try to load and display it
try:
    img = mpimg.imread(sample_path)
    plt.imshow(img)
    plt.title(f"Label: {df['dx'][0]}")
    plt.axis('off')
    plt.show()
    print("✅ Image loaded successfully!")
except Exception as e:
    print(f"❌ Error loading image: {e}")

# save the class order for later use (e.g., when interpreting predictions)
# label_encoder.classes_ gives us the original class names in the order they were encoded 
# (e.g., ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
class_order = list(label_encoder.classes_)


#%%
# ======================================================
# Train/Test Split
# ======================================================

# Split by lesion_id, not by row — HAM10000 has multiple images per lesion.
# A row-level split puts the same lesion in both train and test, leaking patient-specific
# features (skin tone, lighting) into the test set and inflating val_accuracy.
lesion_label_df = df.drop_duplicates(subset='lesion_id')[['lesion_id', 'label']]
train_lesion_ids, test_lesion_ids = train_test_split(
    lesion_label_df['lesion_id'],
    test_size=0.2,
    stratify=lesion_label_df['label'],
    random_state=42
)
train_df_original = df[df['lesion_id'].isin(train_lesion_ids)].reset_index(drop=True)
test_df = df[df['lesion_id'].isin(test_lesion_ids)].reset_index(drop=True)

total_images = len(train_df_original) + len(test_df)
print(f"\nTraining samples: {len(train_df_original)} ({len(train_df_original)/total_images*100:.1f}%)")
print(f"Testing samples:  {len(test_df)} ({len(test_df)/total_images*100:.1f}%)")
print("(Target: ~80% / ~20% — deviation > 5pp warrants investigation)")


#%%
# ======================================================
# Oversample Minority Classes (mild — capped at 3x)
# ======================================================

# cap oversampling at 3x the class's original count to give the model more
# exposure to rare classes without duplicating images enough to cause memorisation
df_list = []
for label in train_df_original['label'].unique():
    df_label = train_df_original[train_df_original['label'] == label]
    target_count = min(
        train_df_original['label'].value_counts().max(),
        len(df_label) * 3
    )
    if len(df_label) < target_count:
        df_label = resample(df_label, replace=True, n_samples=target_count, random_state=42)
    df_list.append(df_label)

train_df_balanced = pd.concat(df_list)
print("\nClass distribution after mild oversampling:")
print(train_df_balanced['label'].value_counts())


#%%
# ======================================================
# Class Weights
# ======================================================

# compute_class_weight calculates a weight for each class inversely proportional to its frequency.
# Rare classes (df, vasc) get high weights so the loss function penalises mistakes on them more.
# Common classes (nv) get low weights so the model isn't dominated by the majority class.
# Use the original (pre-oversampling) distribution so weights reflect the true
# class imbalance. Computing from train_df_balanced produces near-1.0 weights
# for all classes, cancelling out the benefit of class weighting.
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df_original['label']),
    y=train_df_original['label']
)
# Square root softens extreme ratios: the raw balanced weights produce a ~47:1 ratio
# (vasc vs nv), which causes the model to over-predict rare classes with terrible
# precision while misclassifying 57% of nv images. Sqrt brings the ratio to ~7:1,
# preserving the directional imbalance correction without destroying nv recall.
class_weights_softened = np.sqrt(class_weights)
class_weight_dict = dict(enumerate(class_weights_softened))
# Boost mel and akiec — both are clinically important (malignant/precancerous)
# and still under-recalled (mel=0.60, akiec=0.44) in the previous run.
# class order is alphabetical: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6
class_weight_dict[0] *= 1.5  # akiec
class_weight_dict[4] *= 1.5  # mel
print("\nClass weights (sqrt-softened, mel+akiec boosted):")
print(class_weight_dict)


#%%
# ======================================================
# Data Generators
# ======================================================

IMG_SIZE = 224
BATCH_SIZE = 64

# define the data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode='nearest'
)

# for the validation set, we only rescale the pixel values to be between 0 and 1 
test_datagen = ImageDataGenerator(rescale=1./255)

# flow_from_dataframe() creates an iterator that reads images from the paths in the DataFrame, 
# applies the specified augmentations, 
# and yields batches of (image, label) pairs for training and validation.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df_balanced,
    x_col='path',        # the question: where are the images?
    y_col='label_str',   # the answer: what are the labels? 
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=class_order,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',            # the question: where are the images?
    y_col='label_str',       # the answer: what are the labels? 
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    classes=class_order,
    class_mode='categorical',
    shuffle=False
)
# get the number of classes
num_classes = len(train_generator.class_indices)

#%%
# ==========================
# Build CNN Model
# ==========================
# Architecture: VGG-style residual CNN — 4 blocks, 2 convolutions per block,
# filters scaling 64→128→256→512, residual skip connections, expanded dense head.
#
# Design rationale for research:
#
# (1) Two convolutions per block (VGG-style, Simonyan & Zisserman 2015):
#     Two stacked 3×3 convolutions capture the same receptive field as a single 5×5
#     convolution but with fewer parameters and two non-linearities — giving the
#     model more representational depth per spatial level.
#
# (2) Residual (skip) connections (He et al. 2016 — ResNet):
#     Each block adds its input directly to its output: output = F(x) + x.
#     This keeps gradients non-zero through depth (vanishing gradient mitigation)
#     and lets the network learn incremental feature refinements rather than full
#     transformations, which is easier to optimise from limited data.
#     A 1×1 projection convolution is used when filter dimensions change between blocks.
#
# (3) Filters start at 64, scale to 512:
#     Starting at 32 filters for a 224×224 input is undersized — standard practice
#     (VGG, ResNet) starts at 64 and doubles at each pooling step to progressively
#     compress spatial resolution while expanding feature dimensionality.
#
# (4) SpatialDropout2D between blocks (Tompson et al. 2015):
#     Regular Dropout drops individual activations, but adjacent pixels in feature
#     maps are spatially correlated — neighbours fill in the dropped signal.
#     SpatialDropout2D drops entire feature maps, forcing the model to not rely
#     on any single channel, providing stronger regularisation for conv layers.
#
# (5) Dense(512) → Dense(256) classifier head with L2 weight decay:
#     A single Dense(256) layer is a bottleneck when the GAP output has 512 features.
#     Two dense layers with L2 regularisation give the classifier more capacity
#     while regularising against overfitting on the small dataset.

def residual_block(x, filters):
    shortcut = x
    # Project shortcut to match filter dimension when it changes between blocks.
    # A 1×1 convolution without bias followed by BN is the standard projection
    # shortcut from ResNet (option B in He et al. 2016).
    if int(x.shape[-1]) != filters:
        shortcut = Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # First convolution: transform the feature maps
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution: refine the transformed features
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Add shortcut: the network only needs to learn the residual (difference)
    # rather than the full transformation — easier to optimise
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))             # (224, 224,   3)

# Stem (ResNet-style): reduces spatial resolution from 224→56 before residual
# blocks begin. Without this, Block 1 runs two 3×3 convolutions with 64 filters
# at 224×224 — ~3.7B multiply-adds per image per conv (~10 min/epoch).
# The 7×7 stride-2 conv brings 224→112 in one pass; MaxPool brings 112→56.
# All residual blocks then operate at 56×56 or smaller — Block 1 is ~16× cheaper.
x = Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)    # (112, 112,  64)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(3, strides=2, padding='same')(x)                        # ( 56,  56,  64)

x = residual_block(x, 64)                                                # ( 56,  56,  64)
x = MaxPooling2D(2, 2)(x)                                                # ( 28,  28,  64)
x = SpatialDropout2D(0.10)(x)

x = residual_block(x, 128)                                               # ( 28,  28, 128)
x = MaxPooling2D(2, 2)(x)                                                # ( 14,  14, 128)
x = SpatialDropout2D(0.10)(x)

x = residual_block(x, 256)                                               # ( 14,  14, 256)
x = MaxPooling2D(2, 2)(x)                                                # (  7,   7, 256)
x = SpatialDropout2D(0.15)(x)

x = residual_block(x, 512)                                               # (  7,   7, 512)
x = SpatialDropout2D(0.20)(x)

x = GlobalAveragePooling2D()(x)                                          # (512,)

x = Dense(512, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Cosine decay over 80 epochs. LR lowered to 5e-4 (from 1e-3) because the deeper
# residual architecture has more parameters and is sensitive to large initial steps —
# a lower LR stabilises early training without needing an explicit warmup phase.
# 80 epochs gives the model enough time to converge through the larger parameter space.
steps_per_epoch = len(train_generator)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-4,
    decay_steps=80 * steps_per_epoch,
    alpha=1e-6
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
model.summary()
#%%
# ==========================
# Fine-Tune from Checkpoint
# ==========================
# Run this cell INSTEAD of the Build CNN Model cell above when continuing from
# a saved checkpoint. Loads best_model.h5 and recompiles with a lower LR so
# training resumes from epoch-30 weights rather than random initialisation.

from tensorflow.keras.models import load_model as keras_load_model

model = keras_load_model('best_model.h5')
print("✅ Loaded best_model.h5 — resuming from saved weights.")
model.summary()

steps_per_epoch = len(train_generator)
lr_schedule_finetune = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,   # 10x lower than the original run
    decay_steps=30 * steps_per_epoch,
    alpha=1e-7
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_finetune),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

#%%
# ==========================
# Train the Model
# ==========================


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
]

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=60,
    callbacks=callbacks,
    class_weight=class_weight_dict
)


#%%
# ==========================
# Evaluate the Model
# ==========================

# Before evaluating, we reset the test generator to ensure it starts from the beginning of the dataset.
test_generator.reset()
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


#%%
# ==========================
# Save the Model
# ==========================

if os.path.exists('best_model.h5'):
    size_mb = os.path.getsize('best_model.h5') / (1024 * 1024)
    print(f"✅ best_model.h5 confirmed ({size_mb:.1f} MB) — contains best val_loss weights.")
else:
    print("❌ best_model.h5 not found — ModelCheckpoint may not have fired (check val_loss never improved).")


#%%
# ==========================
# Extended Evaluation
# ==========================

# Reset the test generator to ensure it starts from the beginning of the dataset for accurate predictions.
test_generator.reset()
# Get true and predicted labels
true_labels = test_generator.classes
# looks at images and outputs probabilities
pred_probs = model.predict(test_generator) 
# argmax() finds the index of the highest probability for each image, 
# which corresponds to the predicted class label.
# axis = 1 means we look across columns (the class probabilities) for each row (image).
pred_labels = np.argmax(pred_probs, axis=1)

# === 1. Classification Report ===
print("\nClassification Report:")
# generates a text report showing the main classification metrics (precision, recall, f1-score) for each class.
# comparing the true_labels to the pred_labels.
# target_names: Names of the classes for the classification report 
print(classification_report(true_labels, pred_labels, target_names=test_generator.class_indices.keys()))

# === 2. Confusion Matrix ===
# calculates the confusion matrix to visualize the performance of the classification model.
cm = confusion_matrix(true_labels, pred_labels)
# Takes the raw math (cm) and prepares it to be drawn as a pretty, color-coded visual graph
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
# tilts the text at 45 degrees
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# === 3. Training Curves ===
plt.figure() # Creates a blank canvas
# Plots how accuracy on training data changed over 10 epochs
plt.plot(history.history['accuracy'], label='Train Accuracy') 
# Plots how accuracy on the unseen test data changed over 10 epochs
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# This does the exact same thing as the accuracy graph, but for Loss (the error rate).
# If Train Accuracy goes up, but Val Accuracy flatlines or goes down, 
# your model is memorizing the training images instead of actually learning how to spot cancer.
# Ideally, the "Train" line and "Val" line should stay close together as they move up (for accuracy) or down (for loss)
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


#%%
# ==========================
# Grad-CAM Visualization
# ==========================

# Select an image from the validation set
# We need a specific picture to test the visualization on.
img_path = test_df['path'].iloc[0]  # You can change this to view other images
img_size = (IMG_SIZE, IMG_SIZE)

# Load and preprocess image
def load_image(img_path):
    # Reads the image file from hard drive and forces it to be 128x128 pixels.
    # model was strictly built to only accept 128x128 images.
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    # Converts the image into a 3D array (height, width, channels) and adds a 4th dimension for batch size.
    # Neural networks cannot read image files; they can only read arrays of numbers representing colors (Red, Green, Blue).
    img_array = tf.keras.utils.img_to_array(img)
    # Wraps the image array inside another array. This is because the model expects a batch of images, even if it's just one image.
    # adds a 4th dimension for batch size. [128, 128, 3] -> [1, 128, 128, 3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

# gets the preprocessed image array and the original image for visualization
img_array, original_img = load_image(img_path)

# Grad-CAM function
# The original model only outputs probabilities. 
# Grad-CAM needs to see what the last Conv2D layer was "seeing" at the same time as the final prediction
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Creates a new model by specifying what goes in and what comes out
    # a new model that takes the same input as the original model but outputs the activations of the last convolutional layer and the final predictions.
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    # Starts a recording session. 
    # Every mathematical operation that happens inside this block is logged so TensorFlow can later work backwards and calculate slopes (gradients).
    # Gradients tell us "how much did this number affect the final result?" Without the recorder, we would have no idea which pixels were important for the model's decision.
    with tf.GradientTape() as tape:
        # What the last Conv2D layer produced, and what the model predicted
        # how much did conv_outputs contribute to predictions?
        conv_outputs, predictions = grad_model(img_array)
        # automatically picks the class the model is most confident about.
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    # tape.gradient() calculates the slope of the class score with respect to the feature map activations.
    # tape.gradient(input, output) computes the gradient of the output with respect to the input.
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + tf.keras.backend.epsilon())
    return heatmap.numpy()

# Get heatmap and overlay
# last_conv_layer_name = "conv2d_2"  Update this if needed based on model.summary() not optimal

last_conv_layer_name = next(l.name for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D))
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

def overlay_heatmap(heatmap, img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    img = np.array(img)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

overlay_img = overlay_heatmap(heatmap, original_img)

# Show Grad-CAM result
plt.figure(figsize=(6, 6))
plt.imshow(overlay_img)
plt.title("Grad-CAM Heatmap")
plt.axis('off')
plt.show()

# %%
