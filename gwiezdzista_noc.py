import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg19
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

# Open a file dialog to choose the image
Tk().withdraw()
image_path = askopenfilename(title='Select Image')

# Dimensions of the generated picture.
img_nrows, img_ncols = keras.utils.load_img(image_path).size

def preprocess_image(image_path):
    img = keras.utils.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    x = tf.image.resize(x, (img_nrows, img_ncols))
    x = x.numpy()
    x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# Load the trained model
model = keras.models.load_model("trained_model.h5")
model.compile(optimizer='adam', loss='mse')  # Compile the model

# Preprocess the input image
input_image = preprocess_image(image_path)

# Generate the output image
output_image = model.predict(input_image)

# Deprocess the output image
output_image = deprocess_image(output_image)

# Display the output image
plt.imshow(output_image)
plt.axis("off")
plt.show()
