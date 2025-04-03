import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found. Check the file path.")
    

    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def test_model(image_path):
    # Load the trained model
    model = load_model("my_model.keras")
     
    # Preprocess the input image
    img = custom_image(image_path)
    
    # Predict the digit using the model
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predicted digit: {predicted_class[0]}")
    
    # Display the image with its predicted label
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_class[0]}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_model(r'C:\Users\iainc\coding activities\MNIST PROJECT\test_image.png')

