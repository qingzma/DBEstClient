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
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.colors import ListedColormap

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback

# x = np.arange(-1, 1, 0.08)
# y = np.arange(-1, 1, 0.08)
# x, y = np.meshgrid(x, y)
# z = x**2 - y**2
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(x, y, z, alpha=0.2)
# plt.show()

# Real z = (x²-y²) figure


def build_toy_dataset(N):
    # [i for i in range(-1, 1, 0.1) for j in range(-1, 1, 0.1)]#np.float32(np.random.uniform(-1, 1, N))
    x_data = np.array([(2*i/float(N))-1 for i in range(0, N)
                       for j in range(0, N)])
    # np.arange(-1, 1, 0.1)#np.float32(np.random.uniform(-1, 1, N))
    y_data = np.array([(2*j/float(N))-1 for i in range(0, N)
                       for j in range(0, N)])
    z_data = x_data**2-y_data**2
    return x_data, y_data, z_data


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def plot3d(x_data, y_data, z_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data, z_data = build_toy_dataset(100)
    ax.scatter(x_data, y_data, z_data, c='b', marker='o', alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# x_data, y_data, z_data = build_toy_dataset(1000)
# x_data, y_data, z_data = unison_shuffled_copies(x_data, y_data, z_data)
# # plot3d(x_data, y_data, z_data)
x_data = np.linspace(-1,1,1000)
y_data=x_data**2 +x_data+0.01
c = 1  # The number of outputs we want to predict
m = 24  # The number of distributions we want to use in the mixture

# Note: The output size will be (c + 2) * m


def elu_modif(x):
    return tf.nn.elu(x)+1.+1e-8


def log_sum_exp(x, axis=None):
    """ Log-sum-exp trick implementation """
    x_max = backend.max(x,  axis=axis, keepdims=True)
    return backend.log(backend.sum(backend.exp(x - x_max),
                                   axis=axis, keepdims=True))+x_max


def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = tf.keras.backend.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = tf.keras.backend.softmax(tf.keras.backend.clip(alpha, 1e-8, 1.))

    exponent = backend.log(alpha) - .5 * float(c) * backend.log(2 * np.pi) \
        - float(c) * backend.log(sigma) \
        - backend.sum((backend.expand_dims(y_true, 2) - mu)
                      ** 2, axis=1)/(2*(sigma)**2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - backend.mean(log_gauss)
    return res


def mean_log_LaPlace_like(y_true, parameters):
    """Mean Log Laplace Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = backend.reshape(parameters, [-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = backend.softmax(backend.clip(alpha, 1e-2, 1.))

    exponent = backend.log(alpha) - float(c) * backend.log(2 * sigma) \
        - backend.sum(backend.abs(backend.expand_dims(y_true,
                                                      2) - mu), axis=1)/(sigma)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - backend.mean(log_gauss)
    return res
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

inputs = keras.Input(shape=(1,), name='input')
x = keras.layers.Dense(24, activation='relu', name='dense1')(inputs)
x = keras.layers.Dropout(0.25,name='drop1')(x)

FC_mus = keras.layers.Dense(c*m, name='FC_mus')(x)
FC_sigmas = keras.layers.Dense(m, activation=elu_modif, name='FC_sigmas')(x)
FC_alphas = keras.layers.Dense(m,activation='softmax', name='FC_alphas')(x)
output = keras.layers.Concatenate(name="pvec",axis=1)([FC_mus, FC_sigmas, FC_alphas])

model = keras.Model(inputs=inputs, outputs=output)
lossHistory = LossHistory()

from datetime import datetime
start_time = datetime.now()
epoch=1000

model.compile(optimizer='rmsprop',loss=mean_log_Gaussian_like,metrics=['accuracy'])
# out = np.array(list(zip(y_data, z_data)))
# print(out)
train_data = tf.data.Dataset.from_tensor_slices((x_data,y_data))
history = model.fit(train_data, epochs=epoch)#,  verbose=1, batch_size=10000, validation_split=0.1)

end_time = datetime.now()
print() 
print("*********************************  End  *********************************")
print()
print('Duration: {}'.format(end_time - start_time))


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:,i*components:(i+1)*components] for i in range(no_parameters)]

def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector) # Unpack parameter vectors
    
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,       
            scale=sigma))
    
    log_likelihood = gm.log_prob(tf.transpose(y)) # Evaluate log-probability of y
    
    return -tf.reduce_mean(log_likelihood, axis=-1)
class MDN(tf.keras.Model):
    def __init__(self,neurons=100, components=2, dimension=2):
        super(MDN, self).__init__(name="MDN")
        self.neurons = neurons
        self.components = components
        self.dimension = dimension
        self.h1 = keras.layers.Dense(neurons,activation='relu',name='h1')
        self.dropout1 = keras.layers.Dropout(0.25)
        self.h2 = keras.layers.Dense(neurons, activation='relu', name='h2')
        self.dropout2 = keras.layers.Dropout(0.25)

        self.mus = keras.layers.Dense(components, name='FC_mus')
        self.sigmas = keras.layers.Dense(components, activation='nnelu', name='FC_sigmas')
        self.alphas = keras.layers.Dense(components,activation='softmax', name='FC_alphas')
        self.pvec = keras.layers.Concatenate(name="pvec")

    def call(self, inputs):
        x = self.h1(inputs)
        x = self.dropout1(x)
        x = self.h2(x)
        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)
        
        return self.pvec([alpha_v, mu_v, sigma_v])