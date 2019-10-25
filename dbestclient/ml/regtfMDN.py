# from __future__ import absolute_import, division, print_function

# import numpy as np
# import tensorflow as tf

# tf.enable_eager_execution()
# tf.random.set_random_seed(42)
# np.random.seed(42)

# import tensorflow.keras as K

# from tensorflow_probability import distributions as tfd

# from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.datasets import load_boston

# import matplotlib as mpl

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# mpl.rcParams.update(mpl.rcParamsDefault)
# mpl.rcParams.update({'font.size': 12})


# import warnings
# warnings.filterwarnings("always")

# def remove_ax_window(ax):
#     """
#         Remove all axes and tick params in pyplot.
#         Input: ax object.
#     """
#     ax.spines["top"].set_visible(False)    
#     ax.spines["bottom"].set_visible(False)    
#     ax.spines["right"].set_visible(False)    
#     ax.spines["left"].set_visible(False)  
#     ax.tick_params(axis=u'both', which=u'both',length=0)
    
# dpi = 140
# x_size = 8
# y_size = 4
# alt_font_size = 14

# save_figure = False
# use_tb = False



# class MDN(tf.keras.Model):

#     def __init__(self, neurons=100, components = 2):
#         super(MDN, self).__init__(name="MDN")
#         self.neurons = neurons
#         self.components = components
        
#         self.h1 = Dense(neurons, activation="relu", name="h1")
#         self.h2 = Dense(neurons, activation="relu", name="h2")
        
#         self.alphas = Dense(components, activation="softmax", name="alphas")
#         self.mus = Dense(components, name="mus")
#         self.sigmas = Dense(components, activation="nnelu", name="sigmas")
#         self.pvec = Concatenate(name="pvec")
        
#     def call(self, inputs):
#         x = self.h1(inputs)
#         x = self.h2(x)
        
#         alpha_v = self.alphas(x)
#         mu_v = self.mus(x)
#         sigma_v = self.sigmas(x)
        
#         return self.pvec([alpha_v, mu_v, sigma_v])
    
# class DNN(tf.keras.Model):
#     def __init__(self, neurons=100):
#         super(DNN, self).__init__(name="DNN")
#         self.neurons = neurons
        
#         self.h1 = Dense(neurons, activation="relu", name="h1")
#         self.h2 = Dense(neurons, activation="relu", name="h2")
#         self.out = Dense(1, activation="linear", name="out")
        
#     def call(self, inputs):
#         x = self.h1(inputs)
#         x = self.h2(x)
#         return self.out(x)


# def nnelu(input):
#     """ Computes the Non-Negative Exponential Linear Unit
#     """
#     return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

# def slice_parameter_vectors(parameter_vector):
#     """ Returns an unpacked list of paramter vectors.
#     """
#     return [parameter_vector[:,i*components:(i+1)*components] for i in range(no_parameters)]

# def gnll_loss(y, parameter_vector):
#     """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
#     """
#     alpha, mu, sigma = slice_parameter_vectors(parameter_vector) # Unpack parameter vectors
    
#     gm = tfd.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(probs=alpha),
#         components_distribution=tfd.Normal(
#             loc=mu,       
#             scale=sigma))
    
#     log_likelihood = gm.log_prob(tf.transpose(y)) # Evaluate log-probability of y
    
#     return -tf.reduce_mean(log_likelihood, axis=-1)

# tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})


# no_parameters = 3
# components = 1
# neurons = 200

# opt = tf.train.AdamOptimizer(1e-3)

# mon = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')



# mdn = MDN(neurons=neurons, components=components)
# mdn.compile(loss=gnll_loss, optimizer=opt)

# dnn = DNN(neurons=neurons)
# dnn.compile(loss="mse", optimizer=opt)


# samples = int(1e5)

# x_data = np.random.sample(samples)[:, np.newaxis].astype(np.float32)
# y_data = np.add(5*x_data, np.multiply((x_data)**2, np.random.standard_normal(x_data.shape)))

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)


# fig = plt.figure(figsize=(x_size,y_size), dpi=dpi)
# ax = plt.gca()

# ax.set_title(r"$y = 5x + (x^2 * \epsilon)$"+"\n"+r"$\epsilon \backsim \mathcal{N}(0,1)$", fontsize=alt_font_size)
# ax.plot(x_train,y_train, "x",alpha=1., color=sns.color_palette()[0])

# remove_ax_window(ax)
# plt.show()




