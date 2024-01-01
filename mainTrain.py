import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import imutils
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

image_directory = "C:/Users/Computer/Desktop/BrainTumor Classification DL/Aryan/datasets/"

no_tumor_images = os.listdir(image_directory + "no/")
yes_tumor_images = os.listdir(image_directory + "yes/")
dataset = []
label = []

INPUT_SIZE = 124
# print(no_tumor_images)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Tumor", "Tumor"],
                yticklabels=["No Tumor", "Tumor"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_sample_images(x_test, y_true, y_pred_classes, num_samples=5):
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_test[i])
        plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
        plt.axis("off")
    plt.show()

index_to_visualize = 0

# Before preprocessing
original_image = cv2.imread(image_directory + "no/" + no_tumor_images[index_to_visualize])
original_image = Image.fromarray(original_image, "RGB")


for i, image_name in enumerate(no_tumor_images):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(image_directory + "no/" + image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(image_directory + "yes/" + image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

print(dataset)
print(label)


x_train, x_test, y_train, y_test = train_test_split(
    dataset, label, test_size=0.2, random_state=0
)

# Reshape = (n, image_width, image_height, n_channel)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

preprocessed_image = Image.fromarray((x_train[index_to_visualize] * 255).astype(np.uint8), "RGB")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Before Preprocessing")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("After Preprocessing")
plt.imshow(preprocessed_image)
plt.axis("off")

plt.show()

# Model Building
# 64,64,3

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Model Training
history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=15,
                    validation_data=(x_test, y_test), shuffle=True)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(y_true)
# Calculate and print accuracy
accuracy = accuracy_score(y_true, y_pred_classes)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))


# Plot Training History
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(history.history["loss"], label="Train")
ax[0].plot(history.history["val_loss"], label="Validation")
ax[0].set_title("Loss")
ax[0].legend()

ax[1].plot(history.history["accuracy"], label="Train")
ax[1].plot(history.history["val_accuracy"], label="Validation")
ax[1].set_title("Accuracy")
ax[1].legend()
plt.show()

# Plot Confusion Matrix
plot_confusion_matrix(y_true, y_pred_classes)

plot_sample_images(x_test, y_true, y_pred_classes, num_samples=5)

# Plot Sample Images
sample_indices = np.random.choice(len(x_test), 16, replace=False)
sample_images = x_test[sample_indices]
sample_true_labels = y_true[sample_indices]
sample_pred_labels = y_pred_classes[sample_indices]

class_labels=["No Tumor","Tumor"]
plt.figure(figsize=(16,20))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Actual label: {class_labels[sample_true_labels[i]]}\nPredicted label: {class_labels[sample_pred_labels[i]]}")
    plt.axis("off")

plt.show()

IMG_SIZE = (124, 124)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

img = cv2.imread('C:/Users/Computer/Desktop/BrainTumor Classification DL/Aryan/datasets/yes/Y1.jpg')
img = cv2.resize(
            img,
            dsize=IMG_SIZE,
            interpolation=cv2.INTER_CUBIC
        )
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# add contour on the image
img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

# add extreme points
img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)
# crop
ADD_PIXELS = 0
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
# Reshape images for visualization
sample_images = np.squeeze(sample_images)

# Display images using the provided code
iteration = 16
fig = plt.figure(figsize=(10, 10))

plt.figure(figsize=(15,6))
plt.subplot(141)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title('Step 1. Get the original image')
plt.subplot(142)
plt.imshow(img_cnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 2. Find the biggest contour')
plt.subplot(143)
plt.imshow(img_pnt)
plt.xticks([])
plt.yticks([])
plt.title('Step 3. Find the extreme points')
plt.subplot(144)
plt.imshow(new_img)
plt.xticks([])
plt.yticks([])
plt.title('Step 4. Crop the image')
plt.show()

# Threshold image
img = cv2.imread(r'C:\Users\Computer\Desktop\BrainTumor Classification DL\Aryan\datasets\yes\Y1.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')
plt.show()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.19*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(dist_transform, cmap='gray')
plt.title('Distance transform')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sure_fg, cmap='gray')
plt.title('threshold Image')
plt.axis('off')
plt.show()

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(markers)
plt.title('Marker Image')
plt.axis('off')

# marker image with watershed
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,50,0]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(markers)
plt.title('Marker Image after Watershed Segmentation')
plt.axis('off')

#image
cv2.imwrite('img.jpg',img)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

model.save('my_model.keras')

model.summary()

# Resumer de l architecture du model
tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True,
    show_layer_names=True, expand_nested=True)

accuracy = model.evaluate(x_test, y_test)[1]

print("Validation Accuracy: {:.2f}%".format(accuracy * 100))


    
# Load a pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess an image
img_path = r'C:\Users\Computer\Desktop\BrainTumor Classification DL\Aryan\datasets\yes\Y4.jpg'  # Replace with the path to your image
# Load image with Pillow
img = Image.open(img_path)
img = img.resize((224, 224))  # Resize the image to the target size

# Convert the image to a numpy array manually
x = np.asarray(img, dtype='float32')
x = np.expand_dims(x, axis=0)

# Preprocess the input for the VGG16 model
x = preprocess_input(x)

# Get the top predicted class
preds = model.predict(x)
predicted_class = np.argmax(preds[0])
pred_class_name = decode_predictions(preds)[0][0][1]

# Get the output tensor of the last convolutional layer
last_conv_layer = model.get_layer('block5_conv3')

# Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

# Compute the gradient of the top predicted class with respect to the output feature map of the last conv layer
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x)
    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)[0]

# Compute the CAM
cam = np.mean(conv_outputs[0], axis=-1)
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam / cam.max()

# Generate heatmap overlay
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Superimpose the heatmap on the original image
superimposed_img = cv2.addWeighted(
    cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB).astype('float32'), 0.6,
    heatmap.astype('float32'), 0.4, 0
)
# Plot the original image, Grad-CAM heatmap, and the superimposed image
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.title(f'Grad-CAM Heatmap ({pred_class_name})')
plt.imshow(heatmap)

plt.subplot(133)
plt.title(f'Superimposed Image ({pred_class_name})')
plt.imshow(superimposed_img)
plt.show()

from keras.preprocessing.image import ImageDataGenerator
def load_image(image_path):
    # Load an image using OpenCV
    img = cv2.imread(image_path)
    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
img = load_image(r"C:\Users\Computer\Desktop\BrainTumor Classification DL\Aryan\datasets\yes\Y1.jpg")  # Replace with the actual path to your image

# Create an instance of the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example of using data augmentation
img = load_image(r"C:\Users\Computer\Desktop\BrainTumor Classification DL\Aryan\datasets\yes\Y1.jpg")  # Load an image
img = img.reshape((1,) + img.shape)  # Reshape to (1, height, width, channels)

i = 0
for batch in datagen.flow(img, batch_size=1, save_to_dir= r'C:\Users\Computer\Desktop\BrainTumor Classification DL', save_prefix='tumor', save_format='jpg'):
    i += 1
    if i > 20:  # Generate 20 augmented images
        break
# K means

# Ensure that there is enough data for clustering
if len(features) <= 1:
    raise ValueError("Insufficient data for clustering. Number of samples must be greater than 1.")

# Apply PCA with a maximum number of components
max_components = min(len(features), 2)
pca = PCA(n_components=max_components)
features_pca = pca.fit_transform(features)

# Ensure that the number of components is valid
if max_components < 1:
    raise ValueError("Number of components for PCA must be at least 1.")

# Determine the number of clusters dynamically
num_clusters = min(len(features) - 1, 2)  # Ensure num_clusters is less than the number of samples

# Print the number of clusters and number of samples
print(f"Number of clusters: {num_clusters}")
print(f"Number of samples: {len(features)}")

# Apply K-means clustering
if num_clusters >= 1:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_pca)
    # Further processing based on cluster_labels
else:
    raise ValueError("Number of clusters should be at least 1 and less than the number of samples.")
# Plot the K-means clustering results
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    cluster_points = features_pca[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

# Plot centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Centroids')

plt.title('K-means Clustering on Convolutional Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()