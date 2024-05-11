from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__, static_url_path='/static')

# Load the model
loaded_model = tf.keras.models.load_model("brain_tumor_classification_model.keras")

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def decode_predictions(predictions):
    class_labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

def load_explanation(class_name):
    file_path = f"explanations/{class_name}.txt"
    try:
        with open(file_path, 'r') as file:
            explanation = file.read()
        return explanation
    except FileNotFoundError:
        return "Explanation file not found."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the POST request
    file = request.files['file']

    # Save the uploaded image to a temporary location
    file_path = "static/uploaded_image.jpg"
    file.save(file_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(file_path)

    # Make predictions
    predictions = loaded_model.predict(preprocessed_image)

    # Decode predictions
    predicted_class = decode_predictions(predictions)

    # Load explanation for the predicted class
    explanation = load_explanation(predicted_class)

    # Render the result template with prediction and explanation
    return render_template('result.html', prediction=predicted_class, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
