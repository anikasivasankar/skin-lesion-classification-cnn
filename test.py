import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import stats

##Melanocytic nevi (nv)
##Melanoma (mel)
##Benign keratosis-like lesions (bkl)
##Basal cell carcinoma (bcc) 
##Actinic keratoses (akiec)
##Vascular lesions (vas)
##Dermatofibroma (df)

# Read the dataset
skin_df = pd.read_csv('/Users/anikasivasankar/Desktop/skin_cancer/data/HAM10000_metadata.csv')
    
SIZE = 32

# Encoding labels
le = LabelEncoder()
le.fit(skin_df['dx'])
print("Classes: ", list(le.classes_))

skin_df['label'] = le.transform(skin_df["dx"])
print("\nSample of the dataset:")
print(skin_df.sample(10))

# Data distribution visualization
fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type')

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex')

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count', size=12)
ax3.set_title('Localization')

ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red')
ax4.set_title('Age')

plt.tight_layout()
plt.show()

# Balance the dataset
print("\nOriginal class distribution:")
print(skin_df['label'].value_counts())

# Separate classes and resample
n_samples = 500
df_balanced = pd.concat([
    resample(skin_df[skin_df['label'] == i], 
            replace=True, 
            n_samples=n_samples, 
            random_state=42) 
    for i in range(7)
])

print("\nBalanced class distribution:")
print(df_balanced['label'].value_counts())

# Check your actual directory structure and update this path
# Make sure this points to where your .jpg files are actually stored
BASE_DIR = '/Users/anikasivasankar/Desktop/skin_cancer/data/HAM10000_images_part_1'

# Modify the image path creation
image_path = {os.path.splitext(os.path.basename(x))[0]: x
             for x in glob(os.path.join(BASE_DIR, '*.jpg'))}

# Add error handling for image loading
def load_image(path):
    try:
        if path is None:
            return None
        img = Image.open(path)
        return np.asarray(img.resize((SIZE, SIZE)))
    except Exception as e:
        print(f"Error loading image at {path}: {str(e)}")
        return None

# Update the image loading code
df_balanced['path'] = df_balanced['image_id'].map(image_path.get)

# Print some debug information
print("Number of images found:", len(image_path))
print("Sample image paths:", list(image_path.values())[:5])
print("Number of None paths:", df_balanced['path'].isna().sum())

# Load images with error handling
df_balanced['image'] = df_balanced['path'].map(load_image)

# Remove rows where image loading failed
df_balanced = df_balanced.dropna(subset=['image'])

# Display sample images
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize=(4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
plt.show()

# Prepare data for modeling
X = np.asarray(df_balanced['image'].tolist())
X = X/255.  # Normalize pixel values
Y = df_balanced['label']
Y_cat = tf.keras.utils.to_categorical(Y, num_classes=7)

# Split data - using more standard split ratios
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42)
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Define the CNN model
model = tf.keras.Sequential([
    # First Convolutional Block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    # Second Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    # Third Convolutional Block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    # Dense Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# Train model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=25,
    validation_data=(x_valid, y_valid),
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate model
print("\nEvaluating model on test set:")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Save the model
model.save('skin_cancer_model.h5')
print("\nModel saved as 'skin_cancer_model.h5'")