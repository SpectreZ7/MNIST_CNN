import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Block
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_save_model():
    # 1. Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 2. Build the CNN model
    model = build_model()

    # 3. Compile the model using SGD optimizer
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=sgd_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.1)

    # 5. Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # 6. Save the model to disk
    model.save('my_model.keras')



if __name__ == "__main__":
    train_and_save_model()
