import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

input_shape=(224,224)
classes = ['닭가슴살', '닭가슴살샐러드', '닭갈비', '닭강정', '닭고기볶음']

def load_model():
  model = tf.keras.models.load_model('./food_clf/food_classifier_mobilenet.h5')
  return model

def read_image(image_encoded):
  pil_image = Image.open(BytesIO(image_encoded))
  return pil_image

def preprocessing(iamge:Image.Image):
  # Resize the image
  iamge = iamge.resize(input_shape)

  # Convert the image to a numpy array
  img_array = np.array(iamge)

  # Add a batch dimension
  img_array = np.expand_dims(img_array, axis=0)

  # Preprocess the image
  img_array = img_array / 255.0
  return img_array


def pred(img_array:np.ndarray):
  model = load_model()
  pred = model.pred(img_array)
  # Print the predicted class
  predicted_class = classes[np.argmax(pred)]
  return predicted_class