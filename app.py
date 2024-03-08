import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to clean up folder names
def clean_folder_name(folder_name):
    # Implement any cleaning or formatting needed for folder names
    return folder_name.replace("-", " ").title()

# Load ResNet50 pre-trained on ImageNet data (without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Set the correct number of classes in your dataset
num_classes = 99  # Replace with the actual number of classes in your dataset

# Create a new model with a custom top layer for your classification task
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set the path to your dataset
dataset_path = 'dataset'

# Define data generators with data augmentation
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate training dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Generate validation dataset
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Fine-tune the pre-trained model on your dataset
# Increase the number of epochs and experiment with other parameters
model.fit(
    train_generator,
    epochs=20,  # Adjust the number of epochs
    validation_data=validation_generator
)

# Save the embeddings for the dataset
embeddings = []
for img_path in train_generator.filepaths:
    img_array = load_and_preprocess_image(img_path)

    # Get the embedding for each image
    img_embedding = base_model.predict(img_array).reshape(1, -1)
    embeddings.append(img_embedding)

# Save the embeddings to a file
embeddings_array = np.vstack(embeddings)
np.save('dataset_embeddings.npy', embeddings_array)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    capture_option = request.form.get('capture_option')

    top_similar_folders = []  # Initialize with an empty list

    if capture_option == 'upload':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
        else:
            return redirect(url_for('index'))
    elif capture_option == 'camera':
        # Capture an image from the camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        # Save the captured image
        cv2.imwrite('uploads/captured_image.jpg', frame)

        # Release the camera
        cap.release()

        file_path = 'uploads/captured_image.jpg'
    else:
        return redirect(url_for('index'))

    # Load the captured image for prediction
    img_array = load_and_preprocess_image(file_path)

    # Get the embedding for the captured image
    captured_embedding = base_model.predict(img_array).reshape(1, -1)

    # Load dataset embeddings
    dataset_embeddings = np.load('dataset_embeddings.npy')

    # Calculate cosine similarity between the captured image and dataset images
    similarities = cosine_similarity(captured_embedding, dataset_embeddings)

    # Print similarity scores for analysis
    print(f'Similarity Scores: {similarities.flatten()}')

    # Set a lower similarity threshold
    similarity_threshold = 0.1  # Adjust this threshold based on the printed scores

    most_similar_folder_names = []  # Initialize with an empty list

    # Find two most similar folders
    for _ in range(2):
        if np.max(similarities) >= similarity_threshold:
            # If similar, display the most similar folder name from the dataset
            most_similar_index = np.argmax(similarities)
            most_similar_folder_name = os.path.basename(os.path.dirname(train_generator.filepaths[most_similar_index]))

            # Clean up the folder name
            cleaned_folder_name = clean_folder_name(most_similar_folder_name)
            print(f'Most Similar Folder Name: {cleaned_folder_name}')

            # Append to the list
            most_similar_folder_names.append(cleaned_folder_name)

            # Set the similarity score of the found folder to -1 to find the next most similar folder
            similarities[0, most_similar_index] = -1
        else:
            break

    print(f'Two Most Similar Folders: {most_similar_folder_names}')

    return render_template('result.html', top_similar_folders=most_similar_folder_names)

if __name__ == '__main__':
    app.run(debug=True)