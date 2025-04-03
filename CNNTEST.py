import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    

    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img
def predict_number_from_image(image_path, model):
    image = cv2.imread(image_path)
    # greyscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    
    # Using THRESH_BINARY_INV because MNIST digits are white on a black background.
    ret, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    #Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store bounding boxes for each contour that is large enough to be a digit
    bounding_rects = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Filter out too-small contours which may be noise
        if w * h > 100:  # adjust this threshold based on your image
            bounding_rects.append((x, y, w, h))
    
    # sort the numbers from left to right 
    bounding_rects = sorted(bounding_rects, key=lambda b: b[0])
    
    predicted_digits = []
    
    for (x, y, w, h) in bounding_rects:
        # Extract the ROI for the digit
        roi = thresh[y:y+h, x:x+w]
        
        # Resize ROI to match the model's expected input size (28x28)
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize the image (pixel values between 0 and 1)
        roi_resized = roi_resized.astype("float32") / 255.0
        
        # Expand dimensions to add the channel (28,28) -> (28,28,1)
        roi_resized = np.expand_dims(roi_resized, axis=-1)
        
        # Expand dimensions to create a batch of size 1 (1,28,28,1)
        roi_resized = np.expand_dims(roi_resized, axis=0)
        
        prediction = model.predict(roi_resized)
        digit = np.argmax(prediction, axis=1)[0]
        predicted_digits.append(digit)
    
    predicted_number = int("".join(str(d) for d in predicted_digits))
    return predicted_number
def test_model(image_path):
    """
    Loads the trained model, predicts the multi-digit number from the given image,
    prints the predicted number, and displays the original image with the prediction.
    
    This function demonstrates the complete flow from image input to final prediction.
    """
    model = load_model("digit_reader.keras")
    
    predicted_number = predict_number_from_image(image_path, model)
    print("Predicted number:", predicted_number)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(f'Predicted: {predicted_number}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_model(r'multiple_digits2.png')

