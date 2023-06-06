import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg19
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import matplotlib.pyplot as plt

# Wczytaj wcześniej zapisany model
model = keras.models.load_model('trained_model.h5')
model.compile(optimizer='adam', loss='mse')

# Funkcja do transferu stylu
def style_transfer(base_image_path, style_reference_image_path, result_prefix, img_nrows, img_ncols, iterations, model):
    # Weights of the different loss components
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    # Dimensions of the generated picture.
    width, height = keras.utils.load_img(base_image_path).size
    img_ncols = int(width * img_nrows / height)

    def preprocess_image(image_path):
        img = keras.utils.load_img(image_path, target_size=(img_nrows, img_ncols))
        img = keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def deprocess_image(x):
        x = x.reshape((img_nrows, img_ncols, 3))
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def style_loss(style, combination):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_nrows * img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))

    def content_loss(base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(x):
        a = tf.square(x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :])
        b = tf.square(x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    model = vgg19.VGG19(weights="imagenet", include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
    style_layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    content_layer_name = "block5_conv2"

    def compute_loss(combination_image, base_image, style_reference_image):
        input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
        features = model(input_tensor)
        loss = tf.zeros(shape=())

        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(base_image_features, combination_features)

        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(style_layer_names)) * sl

        loss += total_variation_weight * total_variation_loss(combination_image)
        return loss

    @tf.function
    def compute_loss_and_grads(combination_image, base_image, style_reference_image):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96)
    )

    base_image = preprocess_image(base_image_path)
    style_reference_image = preprocess_image(style_reference_image_path)
    combination_image = tf.Variable(preprocess_image(base_image_path))

    iteration_losses = []
    iteration_times = []

    for i in range(1, iterations + 1):
        start_time = time.time()
        loss, grads = compute_loss_and_grads(combination_image, base_image, style_reference_image)
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            end_time = time.time()
            iteration_time = end_time - start_time
            print("Iteration %d: loss=%.2f, time=%.2fs" % (i, loss, iteration_time))
            iteration_losses.append(loss.numpy())
            iteration_times.append(iteration_time)
            img = deprocess_image(combination_image.numpy())
            fname = result_prefix + "_at_iteration_%d.png" % i
            keras.utils.save_img(fname, img)

    plt.plot(range(100, iterations + 1, 100), iteration_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Iteration vs. Loss')
    plt.show()

    img_path = result_prefix + "_at_iteration_%d.png" % iterations
    img = keras.utils.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    model.save("trained_model.h5")


# Utwórz okno dialogowe do wyboru pliku
Tk().withdraw()

# Wybierz plik za pomocą okna dialogowego
base_image_path = askopenfilename(title='Select Base Image')

# Ścieżka do obrazu referencyjnego stylu
style_reference_image_path = keras.utils.get_file("starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg")

# Pozostałe parametry
result_prefix = "image_generated"
img_nrows = 400
img_ncols = 400
iterations = 4000

# Wywołaj funkcję style_transfer z wybranymi parametrami
style_transfer(base_image_path, style_reference_image_path, result_prefix, img_nrows, img_ncols, iterations, model)