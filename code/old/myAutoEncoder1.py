import numpy as np
import tensorflow as tf
import pickle
from myPDEFIND import SpectralDerivs
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def full_network(params):
    """
    Define the full network architecture.
    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.
    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    library_dim = len(params['lib_form'])

    network = {}
    U = tf.compat.v1.placeholder(tf.float32, shape=[None, input_dim], name='U')
    dU = tf.compat.v1.placeholder(tf.float32, shape=[None, input_dim], name='dU')
    z, U_decode, encoder_weights, encoder_biases, decoder_weights, \
    decoder_biases = nonlinear_autoencoder(U,input_dim,latent_dim,params['widths'],activation=activation)
    dz = z_derivative(U, dU, encoder_weights, encoder_biases, activation=activation)
    # Theta = sindy_library_tf(z, latent_dim)
    Theta = sindy_library_tf(z,library_dim)

    if params['coefficient_initialization'] == 'xavier':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', shape=[library_dim, latent_dim],
                                             initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', shape=[library_dim, latent_dim],
                                             initializer=tf.compat.v1.constant_initializer(1.0))
    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', shape=[library_dim, latent_dim],
                                             initializer=tf.compat.v1.initializers.random_normal())

    if params['sequential_thresholding']:
        coefficient_mask = tf.compat.v1.placeholder(tf.float32, shape=[library_dim, latent_dim], name='coefficient_mask')
        sindy_predict = tf.matmul(Theta, coefficient_mask * sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, sindy_coefficients)

    dU_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)


    network['U'] = U
    network['dU'] = dU
    network['z'] = z
    network['dz'] = dz
    network['U_decode'] = U_decode
    network['dU_decode'] = dU_decode
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients
    network['dz_predict'] = sindy_predict
    return network


def define_loss(network, params):
    """
    Create the loss functions.
    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    U = network['U']
    U_decode = network['U_decode']
    dz = network['dz']
    dz_predict = network['dz_predict']
    dU = network['dU']
    dU_decode = network['dU_decode']
    sindy_coefficients = params['coefficient_mask'] * network['sindy_coefficients']
    losses = {}
    losses['decoder'] = tf.reduce_mean(input_tensor=(U - U_decode) ** 2)
    losses['sindy_z'] = tf.reduce_mean(input_tensor=(dz - dz_predict) ** 2)
    losses['sindy_U'] = tf.reduce_mean(input_tensor=(dU - dU_decode) ** 2)
    losses['sindy_regularization'] = tf.reduce_mean(input_tensor=tf.abs(sindy_coefficients))
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_U'] * losses['sindy_U'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_U'] * losses['sindy_U']

    return loss, losses, loss_refinement


def nonlinear_autoencoder(U, input_dim, latent_dim, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.
    Arguments:
    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    z, encoder_weights, encoder_biases = build_network_layers(U, input_dim, latent_dim, widths, activation_function,
                                                              'encoder')
    U_decode, decoder_weights, decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1],
                                                                     activation_function, 'decoder')

    return z, U_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).
    Arguments:
        input - 2D tensorflow array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables
    Returns:
        input - Tensorflow array, output of the network layers (shape is [?,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width = input_dim
    for i, n_units in enumerate(widths):
        W = tf.compat.v1.get_variable(name + '_W' + str(i), shape=[last_width, n_units],
                            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b = tf.compat.v1.get_variable(name + '_b' + str(i), shape=[n_units],
                            initializer=tf.compat.v1.constant_initializer(0.0))
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = tf.compat.v1.get_variable(name + '_W' + str(len(widths)), shape=[last_width, output_dim],
                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
    b = tf.compat.v1.get_variable(name + '_b' + str(len(widths)), shape=[output_dim],
                        initializer=tf.compat.v1.constant_initializer(0.0))
    input = tf.matmul(input, W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases

def sindy_library_tf(z, latent_dim):
    library = [tf.ones(tf.shape(input=z)[0])]
    for i in range(latent_dim):
        library.append(z[:, i])
    return tf.stack(library, axis=1)



def z_derivative(input, dU, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.
    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    dz = dU
    if activation == 'elu':
        for i in range(len(weights) - 1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.minimum(tf.exp(input), 1.0),
                             tf.matmul(dz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights) - 1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.cast(input > 0, dtype=tf.float32), tf.matmul(dz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights) - 1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz = tf.multiply(tf.multiply(input, 1 - input), tf.matmul(dz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
    else:
        for i in range(len(weights) - 1):
            dz = tf.matmul(dz, weights[i])
        dz = tf.matmul(dz, weights[-1])
    return dz

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = full_network(params)
    loss, losses, loss_refinement = define_loss(autoencoder_network, params)
    learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train_op_refinement = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_refinement)
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))

    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    x_norm = np.mean(val_data['U'] ** 2)
    sindy_predict_norm_x = np.mean(val_data['Ut'] ** 2)

    validation_losses = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]

    print('TRAINING')
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(params['max_epochs']):
            for j in range(params['epoch_size'] // params['batch_size']):
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op, feed_dict=train_dict)

            if params['print_progress'] and (i % params['print_frequency'] == 0):
                validation_losses.append(
                    print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

            if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
                params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['sindy_coefficients'])) > params[
                    'coefficient_threshold']
                validation_dict['coefficient_mask:0'] = params['coefficient_mask']
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
                sindy_model_terms.append(np.sum(params['coefficient_mask']))

        print('REFINEMENT')
        for i_refinement in range(params['refinement_epochs']):
            for j in range(params['epoch_size'] // params['batch_size']):
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op_refinement, feed_dict=train_dict)

            if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
                validation_losses.append(
                    print_progress(sess, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm,
                                   sindy_predict_norm_x))

        saver.save(sess, params['data_path'] + params['save_name'])
        pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
        final_losses = sess.run((losses['decoder'], losses['sindy_x'], losses['sindy_z'],
                                 losses['sindy_regularization']),
                                feed_dict=validation_dict)
        sindy_predict_norm_z = np.mean(sess.run(autoencoder_network['dz'], feed_dict=validation_dict) ** 2)
        sindy_coefficients = sess.run(autoencoder_network['sindy_coefficients'], feed_dict={})

        results_dict = {}
        results_dict['num_epochs'] = i
        results_dict['x_norm'] = x_norm
        results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
        results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z
        results_dict['sindy_coefficients'] = sindy_coefficients
        results_dict['loss_decoder'] = final_losses[0]
        results_dict['loss_decoder_sindy'] = final_losses[1]
        results_dict['loss_sindy'] = final_losses[2]
        results_dict['loss_sindy_regularization'] = final_losses[3]
        results_dict['validation_losses'] = np.array(validation_losses)
        results_dict['sindy_model_terms'] = np.array(sindy_model_terms)

        return results_dict


def print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    """
    Print loss function values to keep track of the training progress.
    Arguments:
        sess - the tensorflow session
        i - the training iteration
        loss - tensorflow object representing the total loss function used in training
        losses - tuple of the individual losses that make up the total loss
        train_dict - feed dictionary of training data
        validation_dict - feed dictionary of validation data
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.
    Returns:
        Tuple of losses calculated on the validation set.
    """
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    validation_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=validation_dict)
    print("Epoch %d" % i)
    print("   training loss {0}, {1}".format(training_loss_vals[0],
                                             training_loss_vals[1:]))
    print("   validation loss {0}, {1}".format(validation_loss_vals[0],
                                               validation_loss_vals[1:]))
    decoder_losses = sess.run((losses['decoder'], losses['sindy_x']), feed_dict=validation_dict)
    loss_ratios = (decoder_losses[0] / x_norm, decoder_losses[1] / sindy_predict_norm)
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
    return validation_loss_vals


def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow.
    Arguments:
        data - Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        params - Dictionary object containing model and training parameters. The relevant
        parameters are model_order (which determines whether the SINDy model predicts first or
        second order time derivatives), sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
        model), and learning rate (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
        in to tensorflow. If None, all examples are used.
    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to tensorflow.
    """
    if idxs is None:
        idxs = np.arange(data['U'].shape[0])
    feed_dict = {}
    feed_dict['U:0'] = data['U'][idxs]
    feed_dict['Ut:0'] = data['Ut'][idxs]
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
    feed_dict['learning_rate:0'] = params['learning_rate']
    return feed_dict


