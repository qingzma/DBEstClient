# # Created by Qingzhi Ma at 16/10/2019
# # All right reserved
# # Department of Computer Science
# # the University of Warwick
# # Q.Ma.2@warwick.ac.uk

# from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf

# from tensorflow import keras

# import os, sys
# import skimage
# from skimage import transform
# from skimage.color import rgb2gray
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns



# def test():
#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0

#     # Add a channels dimension
#     x_train = x_train[..., tf.newaxis]
#     x_test = x_test[..., tf.newaxis]

#     train_ds = tf.data.Dataset.from_tensor_slices(
#         (x_train, y_train)).shuffle(10000).batch(32)

#     test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#     class MyModel(Model):
#         def __init__(self):
#             super(MyModel, self).__init__()
#             self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
#             self.flatten = keras.layers.Flatten()
#             self.d1 = keras.layers.Dense(128, activation='relu')
#             self.d2 = keras.layers.Dense(10, activation='softmax')

#         def call(self, x):
#             x = self.conv1(x)
#             x = self.flatten(x)
#             x = self.d1(x)
#             return self.d2(x)

#     # Create an instance of the model
#     model = MyModel()

#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

#     optimizer = tf.keras.optimizers.Adam()

#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#     test_loss = tf.keras.metrics.Mean(name='test_loss')
#     test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#     @tf.function
#     def train_step(images, labels):
#         with tf.GradientTape() as tape:
#             predictions = model(images)
#             loss = loss_object(labels, predictions)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         train_loss(loss)
#         train_accuracy(labels, predictions)

#     @tf.function
#     def test_step(images, labels):
#         predictions = model(images)
#         t_loss = loss_object(labels, predictions)

#         test_loss(t_loss)
#         test_accuracy(labels, predictions)

#     EPOCHS = 5

#     for epoch in range(EPOCHS):
#         for images, labels in train_ds:
#             train_step(images, labels)

#         for test_images, test_labels in test_ds:
#             test_step(test_images, test_labels)

#         template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#         print(template.format(epoch + 1,
#                               train_loss.result(),
#                               train_accuracy.result() * 100,
#                               test_loss.result(),
#                               test_accuracy.result() * 100))

#         # Reset the metrics for the next epoch
#         train_loss.reset_states()
#         train_accuracy.reset_states()
#         test_loss.reset_states()
#         test_accuracy.reset_states()


# def test1():
#     x1 = tf.constant([1, 2, 3, 4])
#     x2 = tf.constant([5, 6, 7, 8])

#     result = tf.multiply(x1, x2)

#     with tf.Session() as sess:
#         output = sess.run(result)
#         print(output)


# def belgium():
#     def load_data(data_directory):
#         directories = [d for d in os.listdir(data_directory)
#                        if os.path.isdir(os.path.join(data_directory, d))]
#         labels = []
#         images = []
#         for d in directories:
#             label_directory = os.path.join(data_directory, d)
#             file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]

#             for f in file_names:
#                 images.append(skimage.data.imread(f))
#                 labels.append(int(d))
#         return images, labels

#     ROOT_PATH = "/home/u1796377/Desktop/"
#     train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
#     test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

#     images, labels = load_data(train_data_directory)

#     # unique_labels = set(labels)
#     # plt.figure(figsize=(15, 15))
#     # i = 1
#     # for label in unique_labels:
#     #     image = images[labels.index(label)]
#     #     plt.subplot(8, 8, i)
#     #     plt.axis('off')
#     #     plt.title('Label {0} ({1})'.format(label, labels.count(label)))
#     #     i += 1
#     #     plt.imshow(image)
#     # plt.show()

#     images28 = [transform.resize(image, (28, 28)) for image in images]
#     images28 = np.array(images28)
#     images28 = rgb2gray(images28)

#     # traffic_signs = [300, 2250, 3650, 4000]
#     # for i in range(len(traffic_signs)):
#     #     plt.subplot(1,4, i+1)
#     #     plt.axis('off')
#     #     plt.imshow(images28[traffic_signs[i]], cmap='gray')
#     #     plt.subplots_adjust(wspace=0.5)
#     # plt.show()

#     # Initialize placeholders
#     x = tf.placeholder(tf.float32, shape=[None, 28, 28])
#     y = tf.placeholder(tf.int32, shape=[None])

#     # Flatten the input data
#     images_flat = tf.contrib.layers.flatten(x)

#     # Fully connected layer
#     logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

#     # loss
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

#     # optimizer
#     train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#     correct_pred = tf.argmax(logits, 1)

#     # accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#     print("images_flat: ", images_flat)
#     print("logits: ", logits)
#     print("loss: ", loss)
#     print("predicted_labels: ", correct_pred)

#     tf.set_random_seed(1234)
#     sess = tf.Session()

#     sess.run(tf.global_variables_initializer())

#     for i in range(201):
#         print('EPOCH', i)
#         _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
#         if i % 10 == 0:
#             print("Loss: ", loss)
#         print('DONE WITH EPOCH')


# def kera1():
#     inputs = tf.keras.Input(shape=(784,), name='img')
#     x = keras.layers.Dense(64, activation='relu')(inputs)
#     x = keras.layers.Dense(64, activation='relu')(x)
#     outputs = keras.layers.Dense(10, activation='softmax')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
#     print(model.summary())
#     # tf.keras.utils.plot_model(model, 'my_first_model.png',show_shapes=True)

#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#     x_train = x_train.reshape(60000, 784).astype('float32') / 255
#     x_test = x_test.reshape(10000, 784).astype('float32') / 255
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=tf.keras.optimizers.RMSprop(),
#                   metrics=['accuracy'])
#     history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
#     test_scores = model.evaluate(x_test, y_test, verbose=2)

#     print('Test loss:', test_scores[0])
#     print('Test accuracy:', test_scores[1])


# def kera2():
#     encoder_input = tf.keras.Input(shape=(28, 28, 1), name='img')
#     x = keras.layers.Conv2D(16, 3, activation='relu')(encoder_input)
#     x = keras.layers.Conv2D(32, 3, activation='relu')(x)
#     x = keras.layers.MaxPool2D(3)(x)
#     x = keras.layers.Conv2D(32, 3, activation='relu')(x)
#     x = keras.layers.Conv2D(16, 3, activation='relu')(x)
#     encoder_output = keras.layers.GlobalMaxPool2D()(x)
#     encoder = keras.Model(encoder_input, encoder_output, name='encoder')
#     encoder.summary()

#     x =  keras.layers.Reshape((4, 4, 1))(encoder_output)
#     x =  keras.layers.Conv2DTranspose(16, 3, activation='relu')(x)
#     x =  keras.layers.Conv2DTranspose(32, 3, activation='relu')(x)
#     x =  keras.layers.UpSampling2D(3)(x)
#     x =  keras.layers.Conv2DTranspose(16, 3, activation='relu')(x)
#     decoder_output = keras.layers.Conv2DTranspose(1, 3, activation='relu')(x)

#     autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
#     autoencoder.summary()


# def mimo():
#     num_tags = 12  # Number of unique issue tags
#     num_words = 10000  # Size of vocabulary obtained when preprocessing text data
#     num_departments = 4  # Number of departments for predictions

#     title_input = tf.keras.Input(shape=(None,), name='title')
#     body_input = tf.keras.Input(shape=(None,), name='body')
#     tags_input = tf.keras.Input(shape=(num_tags,), name='tags')

#     title_features = keras.layers.Embedding(num_words, 64)(title_input)
#     body_features = keras.layers.Embedding(num_words, 64)(body_input)

#     title_features = keras.layers.LSTM(128)(title_features)
#     body_features = keras.layers.LSTM(32)(body_features)

#     x = keras.layers.concatenate([title_features, body_features, tags_input])

#     # Stick a logistic regression for priority prediction on top of the features
#     priority_pred = keras.layers.Dense(1, activation='sigmoid', name='priority')(x)
#     # Stick a department classifier on top of the features
#     department_pred = keras.layers.Dense(num_departments, activation='softmax', name='department')(x)

#     # Instantiate an end-to-end model predicting both priority and department
#     model = keras.Model(inputs=[title_input, body_input, tags_input],
#                         outputs=[priority_pred, department_pred])

#     model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
#                   loss={'priority': 'binary_crossentropy',
#                         'department': 'categorical_crossentropy'},
#                   loss_weights=[1., 0.2])
#     # Dummy input data
#     title_data = np.random.randint(num_words, size=(1280, 10))
#     body_data = np.random.randint(num_words, size=(1280, 100))
#     tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
#     # Dummy target data
#     priority_targets = np.random.random(size=(1280, 1))
#     dept_targets = np.random.randint(2, size=(1280, num_departments))

#     model.fit({'title': title_data, 'body': body_data, 'tags': tags_data},
#               {'priority': priority_targets, 'department': dept_targets},
#               epochs=20,
#               batch_size=32)


# def reg1():
#     dataset_path = keras.utils.get_file("auto-mpg.data",
#                                         "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#     column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
#                     'Acceleration', 'Model Year', 'Origin']
#     raw_data = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t',
#                            sep=" ", skipinitialspace=True)
#     dataset = raw_data.copy()
#     # print(dataset.isna().sum())
#     dataset = dataset.dropna()
#     # print(dataset.isna().sum())

#     origin = dataset.pop('Origin')

#     dataset['USA'] = (origin == 1) * 1.0
#     dataset['Europe'] = (origin == 2) * 1.0
#     dataset['Japan'] =(origin == 3) * 1.0
#     # print(dataset.tail())

#     train_dataset = dataset.sample(frac=0.8, random_state=0)
#     test_dataset = dataset.drop(train_dataset.index)
#     # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#     # plt.show()

#     train_stats = train_dataset.describe()
#     train_stats.pop("MPG")
#     train_stats =train_stats.transpose()
    
#     train_labels = train_dataset.pop('MPG')
#     test_labels = test_dataset.pop('MPG')

#     def norm(x):
#         return (x-train_stats['mean']/train_stats['std'])
#     normed_train_data = norm(train_dataset)
#     normed_test_data = norm(test_dataset)
#     normed_train_data= normed_train_data.values
#     normed_test_data= normed_test_data.values
#     train_labels = train_labels.values
#     test_labels = test_labels.values

    
#     # train_dataset = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
#     # test_dataset = tf.data.Dataset.from_tensor_slices((normed_test_data.values, test_labels.values))
    
#     # BATCH_SIZE=1000
#     # SHUFFLE_BUFFER_SIZE = 100
#     # train_dataset = train_dataset.batch(BATCH_SIZE)
#     # test_dataset = test_dataset.batch(BATCH_SIZE)
    

#     def build_model():
#         model = keras.Sequential([
#             keras.layers.Dense(64, activation='relu', input_shape=[9]),
#             keras.layers.Dense(64, activation='relu'),
#             keras.layers.Dense(1)
#         ])
#         optimizer = keras.optimizers.RMSprop(1e-3)
#         model.compile(loss='mse',optimizer=optimizer, metrics=['mae','mse'])
#         return model
#     model = build_model()
#     model.summary()

#     EPOCHS = 1000

#     # Display training progress by printing a single dot for each completed epoch
#     class PrintDot(keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs):
#             if epoch % 100 == 0: print('')
#             print('.', end='')
#     # The patience parameter is the amount of epochs to check for improvement
#     early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#     history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2,
#                  verbose=0, callbacks=[PrintDot()])

#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch

#     def plot_history(history):
#         hist = pd.DataFrame(history.history)
#         hist['epoch'] = history.epoch

#         plt.figure()
#         plt.xlabel('Epoch')
#         plt.ylabel('Mean Abs Error [MPG]')
#         plt.plot(hist['epoch'], hist['mae'],
#                 label='Train Error')
#         plt.plot(hist['epoch'], hist['val_mae'],
#                 label = 'Val Error')
#         # plt.ylim([0,5])
#         plt.legend()

#         plt.figure()
#         plt.xlabel('Epoch')
#         plt.ylabel('Mean Square Error [$MPG^2$]')
#         plt.plot(hist['epoch'], hist['mse'],
#                 label='Train Error')
#         plt.plot(hist['epoch'], hist['val_mse'],
#                 label = 'Val Error')
#         # plt.ylim([0,20])
#         plt.legend()
#         plt.show()


#     plot_history(history)

#     loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

#     print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

#     test_predictions = model.predict(normed_test_data).flatten()

#     plt.scatter(test_labels, test_predictions)
#     plt.xlabel('True Values [MPG]')
#     plt.ylabel('Predictions [MPG]')
#     plt.axis('equal')
#     plt.axis('square')
#     plt.xlim([0,plt.xlim()[1]])
#     plt.ylim([0,plt.ylim()[1]])
#     _ = plt.plot([-100, 100], [-100, 100])
#     plt.show()

    
    
    
    


# if __name__ == "__main__":
#     reg1()
