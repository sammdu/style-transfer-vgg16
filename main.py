#!/usr/bin/python3

"""
    >>> IMPORTS <<<
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image

# VGG-16 CONVOLUTIONAL NEURAL NETWORK MODEL
import vgg16

# This can change the directory of the model;
# vgg16.data_dir = 'vgg16/'

# This will download the model into ./vgg16
# which is about 553.4MB;
vgg16.maybe_download()


"""
    >>> DATA PROCCESSING <<<
"""

# LOAD IMAGE
def load_image(filename, max_size=None):

    image = PIL.Image.open(filename)

    if max_size is not None:

        factor = max_size / np.max(image.size)          # Calculate re-scaling factor
        size   = np.array(image.size) * factor          # Get new height & width with the factor above
        size   = size.astype(int)                       # Make image size integers for the PIL library
        image  = image.resize(size, PIL.Image.LANCZOS)  # Apply the scale

    return np.float32(image)  # Convert scaled image into a NumPy array

# SAVE MANIPULATED IMAGE AS JPEG
def save_image(image, filename):

    image = np.clip(image, 0.0, 255.0)  # Set bounds for pixel values between 0-255
    image = image.astype(np.uint8)      # Convert image to bytes

    with open(filename, 'wb') as file:  # Write the image-file in jpeg format
        PIL.Image.fromarray(image).save(file, 'jpeg')

# PLOT A LARGE IMAGE
def plot_image_big(image):

    image = np.clip(image, 0.0, 255.0)   # Set bounds for pixel values between 0-255
    image = image.astype(np.uint8)       # Convert pixels to bype

    # Works in Jupyter Notebook, haven't ported into standalone mode, so ignore error for now.
    display(PIL.Image.fromarray(image))  # Convert to PIL image & display it

# PLOT [CONTENT-] [STYLE-] AND [MIXED-] IMAGE
def plot_images(content_image, style_image, mixed_image):

    fig, axes = plt.subplots( 1, 3, figsize=(10, 10) )  # Create figure w/ sub-plots
    fig.subplots_adjust(hspace=0.1, wspace=0.1)         # Adjust spacing

    # ENABLE SMOOTH INTERPOLATION
    smooth = True
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # PLOTTING IMAGES

    # PLOT [CONTENT-IMAGE]
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)  # Pixel values normailzed between 0.0 and 1.0
    ax.set_xlabel("Content")

    # PLOT [STYLE-IMAGE]
    ax = axes.flat[1]
    ax.imshow(style_image / 255.0, interpolation=interpolation)    # Pixel values normailzed between 0.0 and 1.0
    ax.set_xlabel("Style")

    # PLOT [MIXED-IMAGE]
    ax = axes.flat[2]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)    # Pixel values normailzed between 0.0 and 1.0
    ax.set_xlabel("Mixed")

    # REMOVE TICKS FROM ALL PLOTS
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


"""
    >>> LOSS FUNCTIONS <<<
"""

# CALCULATE MEAN SQUARED ERROR FOR TENSORS 'A' AND 'B'
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a-b))

# LOSS FUNCTION FOR [CONTENT-IMAGE]
def create_content_loss(session, model, content_image, layer_ids):
    """
        Parameters:
          [session:] An open TensorFlow session for running the model's graph.
            [model:] The model, i.e. an instance of the VGG16-class.
    [content_image:] Numpy float array with the content-image.
        [layer_ids:] List of integer id's for the layers to use in the model.
    """

    # Create a dictionary feed with [content_image]
    feed_dict = model.create_feed_dict(image=content_image)

    # Get references to the tensors for the given layers
    layers    = model.get_layer_tensors(layer_ids)

    # Calculate the output values of the layers when feeding [content_image]
    values    = session.run(layers, feed_dict=feed_dict)

    # Set the model's graph as default
    with model.graph.as_default():

        # Initialize an empty list of losses
        layer_losses = []

        # Feed back layer's loss and compare to the loss of [content_image]
        for value, layer in zip(values, layers):
            value_const = tf.constant(value)                      # Make sure output value is a constant
            loss        = mean_squared_error(layer, value_const)  # The actual loss function
            layer_losses.append(loss)                             # Add the loss function for this layer to the list

        # Calculate total loss, which is the average loss of all layers
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

# COPY ACTIVATION PATTERN FROM STYLE LAYERS TO [MIXED-IMAGE] VIA GRAM-MATRIX
def gram_matrix(tensor):

    shape = tensor.get_shape()

    # Get the number of feature channels for the input tensor
    num_channels = int(shape[3])

    # Re-shape the tensor so it's a 2-dimention matrix,
    # a.k.a. flattening the feature channels.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    # Calculate the Gram Matrix as the matrix product of the tensor with itself;
    # This calculates the dot products of all combinations of the feature channels.
    gram = tf.matmul( tf.transpose(matrix), matrix )

    return gram

# LOSS FUNCTION FOR [STYLE-IMAGE]
# (This calculates the mean squared error for the Gram Matrices instead of the raw outputs)
def create_style_loss(session, model, style_image, layer_ids):
    """
      Parameters:
        [session:] An open TensorFlow session for running the model's graph.
          [model:] The model, e.g. an instance of the VGG16-class.
    [style_image:] Numpy float array with the style-image.
      [layer_ids:] List of integer id's for the layers to use in the model.
    """

    # Create a dictionary feed with [STYLE-IMAGE]
    feed_dict = model.create_feed_dict(image=style_image)

    # Get references to the tensors for the given layers
    layers    = model.get_layer_tensors(layer_ids)

    # Set the model's graph as default
    with model.graph.as_default():

        gram_layers  = [ gram_matrix(layer) for layer in layers ]      # Construct the calculation for Gram Matrices
        values       = session.run(gram_layers, feed_dict=feed_dict)   # Calculate values of Gram Matrices when feeding [style-image]
        layer_losses = []                                              # Initialize an empty list of losses

        # Feed back layer's loss and compare to the loss of [style-image]
        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)                           # Make sure output value is a constant
            loss        = mean_squared_error(gram_layer, value_const)  # Actual loss function, mean squared error between Gram Matrix values
            layer_losses.append(loss)                                  # Add the loss function for this layer to the list

        # Calculate total loss, which is the average loss of all layers
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

# LOSS FUNCTION FOR DENOISING THE [MIXED-IMAGE] (TOTAL VARIATION DENOISING)
def create_denoise_loss(model):

    loss = tf.reduce_sum( tf.abs(model.input[:, 1:, :, :] - model.input[:, :-1, :, :]) ) + \
           tf.reduce_sum( tf.abs(model.input[:, :, 1:, :] - model.input[:, :, :-1, :]) )

    return loss


"""
    >>> STYLE-TRANSFER ALGORITHM <<<
"""

# PERFORM GRADIENT DESCENT ON THE LOSS FUNCTION
# (The weight values are relative to each other)
def style_transfer(content_image,
                   style_image,
                   content_layer_ids,
                   style_layer_ids,
                   weight_content=1.5,
                     weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120,
                        step_size=10.0):
    """
            Parameters:
        [content_image:] Numpy 3-dim float-array with the content-image.
          [style_image:] Numpy 3-dim float-array with the style-image.
    [content_layer_ids:] List of integers identifying the content-layers.
      [style_layer_ids:] List of integers identifying the style-layers.
       [weight_content:] Weight for the content-loss-function.
         [weight_style:] Weight for the style-loss-function.
       [weight_denoise:] Weight for the denoising-loss-function.
       [num_iterations:] Number of optimization iterations to perform.
            [step_size:] Step-size for the gradient in each iteration.
    """

    # Create an instance of the VGG-16 model everytime this function is called,
    # so we can use RAM more efficiently.
    model = vgg16.VGG16()

    # Create an interactive TensorFlow session
    session = tf.InteractiveSession(graph=model.graph)

    # Print the names of the content layers
    print("Content Layers: ")
    print(model.get_layer_names(content_layer_ids))
    print()

    # Print the names of the style layers
    print("Style Layers: ")
    print(model.get_layer_names(style_layer_ids))
    print()

    # Create loss function between the [content-layers] and [content-image]
    loss_content = create_content_loss(session=session,
                                         model=model,
                                 content_image=content_image,
                                     layer_ids=content_layer_ids)

    # Create loss function between the [style-layers] and [style-image]
    loss_style   =   create_style_loss(session=session,
                                         model=model,
                                   style_image=style_image,
                                     layer_ids=style_layer_ids)

    # Create loss function for denoising the [mixed-image]
    loss_denoise = create_denoise_loss(model)

    # Create TensorFlow variables adjusting the values of the loss functions
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style   = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustment values for the loss function
    session.run([adj_content.initializer,
                   adj_style.initializer,
                 adj_denoise.initializer])

    # Create TensorFlow operations to update the adjustment values
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style   =   adj_style.assign(1.0 / (loss_style   + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    # Ultimate weighted loss function that generates the [mixed-image]
    loss_combined = weight_content * adj_content * loss_content + \
                    weight_style   * adj_style   * loss_style + \
                    weight_denoise * adj_denoise * loss_denoise

    # Come up with a gradient for the ultimate loss function regaring the input image
    gradient = tf.gradients(loss_combined, model.input)

    # List of operation to run every epoch
    run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]

    # [mixed-image] is initialized with random noise
    mixed_image = np.random.rand(*content_image.shape) + 128

    # Loop for epochs
    for i in range(num_iterations):

        # Create dictionary feed with [mixed-image]
        feed_dict = model.create_feed_dict(image=mixed_image)

        # Calculate gradient with TensorFlow, then update adjustment values
        grad, adj_content_val, adj_style_val, adj_denoise_val \
            = session.run(run_list, feed_dict=feed_dict)

        # Reduce the dimentions of the gradient
        grad = np.squeeze(grad)

        # Scale the step-size according to the gradient values
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image according to the gradient
        mixed_image -= grad * step_size_scaled

        # Ensure the image has valid pixel values between 0 and 255
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        # Print a little progress indicator with dots
        print(". ", end="")

        # Display status every 20 epochs, as well as the last epoch
        if (i % 20 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration: ", i)
            # Print adjustment weights for loss functions
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print( msg.format(adj_content_val, adj_style_val, adj_denoise_val) )
            # Plot all three images
            plot_images(content_image=content_image,
                          style_image=style_image,
                          mixed_image=mixed_image)

    print()
    print("Final Image: ")
    print()
    plot_image_big(mixed_image)

    session.close()     # Close TensorFlow session

    return mixed_image  # Return the [mixed-image]


"""
    >>> Execute Program <<<
"""

# LOAD CONTENT IMAGE
content_filename = 'images/male.jpg'
content_image = load_image(content_filename, max_size=None)

# LOAD STYLE IMAGE
style_filename = 'styles/abstract-1.jpg'
style_image = load_image(style_filename, max_size=300)

# DEFINE A LAYER INDEX IN THE VGG-16 MODEL FOR THE CONTENT IMAGE, DEFAULT IS INDEX-4 (FIFTH LAYER)
content_layer_ids = [4]

# DEFINE A LAYER INDEX IN THE VGG-16 MODEL FOR THE STYLE IMAGE, DEFAULT IS 13
style_layer_ids = list(range(13))

# PERFORM STYLE TRANSFER!!!
img = style_transfer(content_image=content_image,
                       style_image=style_image,
                 content_layer_ids=content_layer_ids,
                   style_layer_ids=style_layer_ids,
                    weight_content=1.5,
                      weight_style=12.0,
                    weight_denoise=0.3,
                    num_iterations=100,
                         step_size=10.0)
