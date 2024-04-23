# A classical NN; adapted from https://www.tensorflow.org/tutorials/keras/classification/
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterSampler
from tensorflow import keras
from tensorflow.python.client import device_lib
import numpy as np


#trying to use GPU, if available for faster training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())


#function which loads all images from pc_parts folder and preprocess them
def load_images_from_folder(folder, mode='train'):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                #target size of each image is 128px and rgb mode for better identification of class
                img = keras.preprocessing.image.load_img(img_path, target_size=(128, 128), color_mode='rgb')
                img_array = keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
    if mode == 'train':
        return np.array(images), np.array(labels)
    elif mode == 'test':
        return np.array(images), np.array(labels)



dataset_folder = 'pc_parts'
train_images, train_labels = load_images_from_folder(dataset_folder, mode='train')

# class_images = {class_idx: None for class_idx in np.unique(train_labels)}
#
# for image, label in zip(train_images, train_labels):
#     if class_images[label] is None:
#         class_images[label] = image
#
# fig, axes = plt.subplots(1, len(class_images), figsize=(len(class_images) * 3, 3))
# for ax, (class_idx, image) in zip(axes, class_images.items()):
#     ax.imshow(image)
#     ax.axis('off')
#     ax.set_title(f"Class {class_idx}")
# plt.show()

# init test images and labels to test the model
test_images, test_labels = load_images_from_folder(dataset_folder, mode='test')

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

output_classes = len(np.unique(train_labels))
print("Output classes",output_classes)


#hyper-parameters for extra functionality task C
param_grid = {
    "NUM_FILTERS_CONV1": [32, 64, 128],
    "NUM_FILTERS_CONV2": [64, 128, 256],
    "DROPOUT_RATE": [0.25, 0.5, 0.75],
    "LEARNING_RATE": [0.001, 0.0001, 0.00001],
    "EPOCHS": [20, 30, 40],
    "BATCH_SIZE": [64, 128, 256]
}

# Number of configurations to try
n_configs = 5

# hyper-parameters randomly assigned in the model
random_search = list(ParameterSampler(param_grid, n_configs, random_state=42))

# five different version of model will be used.
for i, config in enumerate(random_search):
    print(f"\nConfiguration {i + 1}/{n_configs}: {config}")

    # Build the model
    model = keras.Sequential([
        keras.Input(shape=(128, 128, 3)),
        keras.layers.Conv2D(filters=config["NUM_FILTERS_CONV1"], kernel_size=(3, 3), activation="relu"), #for final model: filters = 128
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=config["NUM_FILTERS_CONV2"], kernel_size=(3, 3), activation="relu"), #for final model: filters = 256
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(config["DROPOUT_RATE"]), #for final model: dropout_rate = 0.25
        keras.layers.Dense(output_classes, activation="softmax")
    ])

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"]) #for final model: learnin rate = 0.25
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"], verbose=0) #for final model: epoch = 40, batch_size = 64

    # Evaluate the trained model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test accuracy:', test_acc)  #for final model: accuracy = Test accuracy: 0.9737098217010498, loss = 0.0861

    try:
        model.save(f'my_model_w_hyperparameters{i+1}.h5')
        print("Model is saved!")
    except Exception as e:
        print(f"Error with saving model {e}")