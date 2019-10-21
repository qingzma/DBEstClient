# Created by Qingzhi Ma
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy

class MDN():
    def __init__(self, n_features=1, n_kernels=20):
        # tf.executing_eagerly()
        self.n_features=n_features
        self.n_kernels=n_kernels
        self.model = None
        self.optimizer = None


    def network(self):
        # network
        input = tf.keras.Input(shape=(self.n_features,))
        layer = tf.keras.layers.Dense(50, activation='tanh', name='baselayer')(input)
        mu = tf.keras.layers.Dense(self.n_features*self.n_kernels, activation=None, name='mean_layer')(layer)
        # variance (should be greater than 0 so we exponentiate it)
        var_layer = tf.keras.layers.Dense(self.n_kernels, activation=None, name='dense_var_layer')(layer)
        var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(self.n_kernels,), name='variance_layer')(var_layer)

        # mixing coefficient should sum to 1.0
        pi = tf.keras.layers.Dense(self.n_kernels, activation='softmax', name='pi_layer')(layer)

        self.model = tf.keras.models.Model(input, [pi, mu, var])
        self.optimizer = tf.keras.optimizers.Adam()

    def fit(self, X, Y, epochs=6000, b_show_loss_curve=True):
        # Use Dataset API to load numpy data (load, shuffle, set batch size)
        losses = []
        EPOCHS = epochs
        N = X.shape[0]
        dataset = tf.data.Dataset \
            .from_tensor_slices((X, Y)) \
            .shuffle(N).batch(N)
        print_every = int(0.1 * EPOCHS)

        # Start training
        print('Print every {} epochs'.format(print_every))
        for i in range(EPOCHS):
            for train_x, train_y in dataset:
                loss = train_step(self.model, self.optimizer, train_x, train_y)
                losses.append(loss)
            if i % print_every == 0:
                print('Epoch {}/{}: loss {}'.format(i, EPOCHS, losses[-1]))
        
        if b_show_loss_curve:
            # Let's plot the training loss
            plt.plot(range(len(losses)), losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training loss')
            plt.show()


    def predict(self, Xs, b_return_avg=True):
        pi_vals, mu_vals, var_vals = self.model.predict(Xs)

        sampled_predictions = sample_predictions(pi_vals, mu_vals, var_vals, self.n_features, 10)
        
        if b_return_avg:
            return np.mean(sampled_predictions, axis=1)
        else:
            return sampled_predictions
        

    def save(self, path):
        print("Saving model " + path.split('/')[-1] +'...')
        self.model.save(path)
        print('Saved.')
    
    def load(self, path):
        print("Loading model " + path.split('/')[-1] +'...')
        self.model = tf.keras.models.load_model(path)
        print("Loaded")




def create_book_example(n=1000):
    # sample uniformly over the interval (0,1)
    X = np.random.uniform(0., 1., (n,1)).astype(np.float32)    
    # target values 
    y = X + 0.3 * np.sin(2 * np.pi * X) + np.random.uniform(-0.1, 0.1, size=(n,1)).astype(np.float32)
    # test data
    x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)    
    return X, y, x_test

# Build model
def get_model(h=16, lr=0.001):
    input = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Dense(h, activation='tanh')(input)
    x = tf.keras.layers.Dense(1, activation=None)(x)

    model = tf.keras.models.Model(input, x)
    # Use Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse', metrics=['acc'])
#     model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


def run():
    # Are we executing eagerly
    tf.executing_eagerly()
    # Plot data (x and y)
    X, y, x_test = create_book_example(n=4000)
    # plt.plot(X, y, 'ro', alpha=0.04)
    # plt.show()
    # print(tf.__version__)
    # print(tf.test.is_gpu_available())

    # Load and train the network
    model = get_model(h=50)
    epochs=1000
    # Change verbosity (e.g verbose=1) to view the training progress
    history = model.fit(X, y, epochs=epochs, verbose=0)
    print('Final loss: {}'.format(history.history['loss'][-1]))
    # Plot the loss history
    plt.plot(range(epochs), history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training')
    plt.show()


    y_test = model.predict(x_test)
    plt.plot(X, y, 'ro', alpha=0.05, label='train')
    plt.plot(x_test, y_test, 'bo', alpha=0.3, label='test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Plot train and test data')
    plt.show()

def mdn():
    # Are we executing eagerly
    # tf.executing_eagerly()
    # Plot data (x and y)
    X, y, x_test = create_book_example(n=4000)
    # single input feature
    l = 1

    # number of gaussians to represent the multimodal distribution
    k = 60

    # network

    input = tf.keras.Input(shape=(l,))
    layer = tf.keras.layers.Dense(50, activation='tanh', name='baselayer')(input)
    mu = tf.keras.layers.Dense(l*k, activation=None, name='mean_layer')(layer)
    # variance (should be greater than 0 so we exponentiate it)
    var_layer = tf.keras.layers.Dense(k, activation=None, name='dense_var_layer')(layer)
    var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), output_shape=(k,), name='variance_layer')(var_layer)

    # mixing coefficient should sum to 1.0
    pi = tf.keras.layers.Dense(k, activation='softmax', name='pi_layer')(layer)

    model = tf.keras.models.Model(input, [pi, mu, var])
    optimizer = tf.keras.optimizers.Adam()

    model.summary()
    # tf.keras.utils.plot_model(model,'mdn.png')

    flipped_x = deepcopy(y)
    flipped_y = deepcopy(X)

    # Use Dataset API to load numpy data (load, shuffle, set batch size)
    N = flipped_x.shape[0]
    dataset = tf.data.Dataset \
        .from_tensor_slices((flipped_x, flipped_y)) \
        .shuffle(N).batch(N)

    
    losses = []
    EPOCHS = 6000
    print_every = int(0.1 * EPOCHS)

    # Define model and optimizer
    model = tf.keras.models.Model(input, [pi, mu, var])
    optimizer = tf.keras.optimizers.Adam()

    # Start training
    print('Print every {} epochs'.format(print_every))
    for i in range(EPOCHS):
        for train_x, train_y in dataset:
            loss = train_step(model, optimizer, train_x, train_y)
            losses.append(loss)
        if i % print_every == 0:
            print('Epoch {}/{}: loss {}'.format(i, EPOCHS, losses[-1]))
    
    # Let's plot the training loss
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.show()


    # Get predictions
    pi_vals, mu_vals, var_vals = model.predict(x_test)
    pi_vals.shape, mu_vals.shape, var_vals.shape

    # Get mean of max(mixing coefficient) of each row
    preds = approx_conditional_mode(pi_vals, var_vals, mu_vals, l)

    # Plot along with training data
    fig = plt.figure(figsize=(8, 8))
    plt.plot(flipped_x, flipped_y, 'ro')
    plt.plot(x_test, preds, 'g.')
    # plt.plot(flipped_x, preds2, 'b.')
    plt.show()

    sampled_predictions = sample_predictions(pi_vals, mu_vals, var_vals,l, 10)

    # Plot the predictions along with the flipped data 
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(6, 6))
    plt.plot(flipped_x, flipped_y, 'ro', label='train')
    for i in range(sampled_predictions.shape[1]):
        plt.plot(x_test, sampled_predictions[:, i], 'g.', alpha=0.3, label='predicted')
    patches = [
    mpatches.Patch(color='green', label='Training'),
    mpatches.Patch(color='red', label='Predicted')
    ]

    plt.legend(handles=patches)
    plt.show()

def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu)**2
    value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
    return value


def mdn_loss(y_true, pi, mu, var):
    """MDN Loss Function
    The eager mode in tensorflow 2.0 makes is extremely easy to write 
    functions like these. It feels a lot more pythonic to me.
    """
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)



def approx_conditional_mode(pi, var, mu,l):
    """Approx conditional mode
    Because the conditional mode for MDN does not have simple analytical
    solution, an alternative is to take mean of most probable component 
    at each value of x (PRML, page 277)
    """
    n, k = pi.shape
    out = np.zeros((n, l))
    # Get the index of max pi value for each row
    max_component = tf.argmax(pi, axis=1)
    for i in range(n):
        # The mean value for this index will be used
        mc = max_component[i].numpy()
        for j in range(l):
            out[i, j] = mu[i, mc*(l+j)]
    return out

def sample_predictions(pi_vals, mu_vals, var_vals,l, samples=10):
    n, k = pi_vals.shape
    # print('shape: ', n, k, l)
    # place holder to store the y value for each sample of each row
    out = np.zeros((n, samples, l))
    for i in range(n):
        for j in range(samples):
            # for each sample, use pi/probs to sample the index
            # that will be used to pick up the mu and var values
            idx = np.random.choice(range(k), p=pi_vals[i])
            for li in range(l):
                # Draw random sample from gaussian distribution
                out[i,j,li] = np.random.normal(mu_vals[i, idx*(li+l)], np.sqrt(var_vals[i, idx]))
    return out

@tf.function
def train_step(model, optimizer, train_x, train_y):
    # GradientTape: Trace operations to compute gradients
    with tf.GradientTape() as tape:
        pi_, mu_, var_ = model(train_x, training=True)
        # calculate loss
        loss = mdn_loss(train_y, pi_, mu_, var_)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


if __name__=="__main__":
    mdn1= MDN()

    X, y, x_test = create_book_example(n=100)
    mdn1.network()
    mdn1.fit(X, y,epochs=1000,b_show_loss_curve=False)
    print(X[:2])
    mdn1.predict(X[:2])
    mdn1.save("/home/u1796377/Desktop/tfmodel")
    mdn1.load('/home/u1796377/Desktop/tfmodel')

    mdn1.predict(X[:2])

    # mdn1.save("/home/u1796377/Desktop/model/")
    



    