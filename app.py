from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import traceback
# from tensorflow.keras.optimizers import Adam

# Load your pre-trained model with the custom learning rate
# learning_rate=0.0006
# model = tf.keras.models.load_model('model.h5', custom_objects={'Adam': tf.keras.optimizers.Adam(learning_rate=0.0006)})

model = tf.keras.models.load_model('../skin/model.h5',compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),loss='categorical_crossentropy')

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your pre-trained model (assuming it's in TensorFlow/Keras format)
model = tf.keras.models.load_model('../skin/model.h5',compile=False)

# Class names (should match the labels the model was trained on)
class_names = ('Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)','Basal Cell Carcinoma','Benign Keratosis','Dermatofibroma','Melanoma','Melanocytic Nevi','Vascular skin lesion')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_infection():
    try:
        # Get the uploaded image
        image = request.files['image']

        # Save the image to the uploads directory
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Open and preprocess the image
        img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        img = img.resize((224, 224))  # Resize to the input size expected by the model
        img_array = np.array(img) / 255.0  # Normalize image to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class of the image
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        lesion_type = class_names[predicted_class]

        return jsonify({'lesion_type': lesion_type})

    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())  # Debugging: Print the full traceback to track the error
        return jsonify({'error': 'Error predicting lesion type'})

if __name__ == '__main__':
    app.run(debug=True)   