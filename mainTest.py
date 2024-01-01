import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os

model_file_path = r'C:\Users\Computer\Desktop\BrainTumor Classification DL\Saumitra\my_model.keras'
if not os.path.exists(model_file_path):
    print(f"Error: Model file not found at {model_file_path}")
else:
    model = load_model(model_file_path)

    # Load the image
    image = cv2.imread(r'C:\Users\Computer\Desktop\BrainTumor Classification DL\Saumitra\uploads\pred0.jpg')

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Image not loaded.")
    else:
        img = Image.fromarray(image)

        img = img.resize((124, 124))

        img = np.array(img)

        input_img = np.expand_dims(img, axis=0)

        result = model.predict(input_img)
