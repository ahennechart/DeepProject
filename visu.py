# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:57:31 2017

@author: kacem
"""


from __future__ import print_function

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import scipy
import argparse
from keras.models import model_from_json
from keras.applications import inception_v3
from keras import backend as K

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
#parser.add_argument('base_image_path', metavar='base', type=str,
#                    help='Path to the image to transform.')
#parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = 'img_test.png'
result_prefix = 'img'


# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}


# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
weights=loaded_model.load_weights("model.h5")
print("Loaded model from disk")



def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(0)

# Build the InceptionV3 network with our placeholder.
# The model will be loaded with pre-trained ImageNet weights.
#loaded_model = inception_v3.InceptionV3(weights='imagenet',
#                                 include_top=False)
dream = loaded_model.input
print('Model loaded.')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in loaded_model.layers])

# Define the loss.
loss = K.variable(0.)
for layer_name in settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


"""Process:
- Load the original image.
- Define a number of processing scales (i.e. image shapes),
    from smallest to largest.
- Resize the original image to the smallest scale.
- For every scale, starting with the smallest (i.e. current one):
    - Run gradient ascent
    - Upscale image to the next scale
    - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size.
To obtain the detail lost during upscaling, we simply
take the original image, shrink it down, upscale it,
and compare the result to the (resized) original image.
"""


# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 10.

img = preprocess_image(base_image_path)
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img(img, fname=result_prefix + '.png')


#
#    def make_mosaic(im, nrows, ncols, border=1):
#        """From http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
#        """
#        import numpy.ma as ma
#
#        nimgs = len(im)
#        imshape = im[0].shape
#        
#        mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
#                                ncols * imshape[1] + (ncols - 1) * border),
#                                dtype=np.float32)
#        
#        paddedh = imshape[0] + border
#        paddedw = imshape[1] + border
#        print(im)
#        for i in range(nimgs):
#            
#            row = int(np.floor(i / ncols))
#            col = i % ncols
#            
#            mosaic[row * paddedh:row * paddedh + imshape[0],
#                col * paddedw:col * paddedw + imshape[1]] = im[i]
#            
#        return mosaic
#
#
#    def get_weights_mosaic(model, layer_id, n=64):
#        """
#        """
#        
#        # Get Keras layer
#        layer = model.layers[layer_id]
#
#        # Check if this layer has weight values
#        if not hasattr(layer, "W"):
#            raise Exception("The layer {} of type {} does not have weights.".format(layer.name,
#                                                            layer.__class__.__name__))
#            
#        weights = layer.W.get_value()
#        
#        # For now we only handle Conv layer like with 4 dimensions
#        if weights.ndim != 4:
#            raise Exception("The layer {} has {} dimensions which is not supported.".format(layer.name, weights.ndim))
#        
#        # n define the maximum number of weights to display
#        if weights.shape[0] < n:
#            n = weights.shape[0]
#            
#        # Create the mosaic of weights
#        nrows = int(np.round(np.sqrt(n)))
#        ncols = int(nrows)
#
#        if nrows ** 2 < n:
#            ncols +=1
#
#        mosaic = make_mosaic(weights[:n, 0], nrows, ncols, border=1)
#        
#        return mosaic
#
#
#    def plot_weights(model, layer_id, n=64, ax=None, **kwargs):
#        """Plot the weights of a specific layer. ndim must be 4.
#        """
#        import matplotlib.pyplot as plt
#        
#        # Set default matplotlib parameters
#        if not 'interpolation' in kwargs.keys():
#            kwargs['interpolation'] = "none"
#            
#        if not 'cmap' in kwargs.keys():
#            kwargs['cmap'] = "gray"
#        
#        layer = model.layers[layer_id]
#        
#        mosaic = get_weights_mosaic(model, layer_id, n=64)
#        
#        # Plot the mosaic
#        if not ax:
#            fig = plt.figure()
#            ax = plt.subplot()
#        
#        im = ax.imshow(mosaic, **kwargs)
#        ax.set_title("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))
#        
#        plt.colorbar(im, ax=ax)
#        
#        return ax
#
#
#    def plot_all_weights(model, n=64, **kwargs):
#        """
#        """
#        import matplotlib.pyplot as plt
#        from mpl_toolkits.axes_grid1 import make_axes_locatable
#        
#        # Set default matplotlib parameters
#        if not 'interpolation' in kwargs.keys():
#            kwargs['interpolation'] = "none"
#
#        if not 'cmap' in kwargs.keys():
#            kwargs['cmap'] = "gray"
#
#        layers_to_show = []
#
#        for i, layer in enumerate(model.layers):
#            if hasattr(layer, "W"):
#                weights = layer.W.get_value()
#                if weights.ndim == 4:
#                    layers_to_show.append((i, layer))
#
#
#        fig = plt.figure(figsize=(15, 15))
#        
#        n_mosaic = len(layers_to_show)
#        nrows = int(np.round(np.sqrt(n_mosaic)))
#        ncols = int(nrows)
#
#        if nrows ** 2 < n_mosaic:
#            ncols +=1
#
#        for i, (layer_id, layer) in enumerate(layers_to_show):
#
#            mosaic = get_weights_mosaic(model, layer_id=layer_id, n=n)
#
#            ax = fig.add_subplot(nrows, ncols, i+1)
#            
#            im = ax.imshow(mosaic, **kwargs)
#            ax.set_title("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))
#
#            divider = make_axes_locatable(ax)
#            cax = divider.append_axes("right", size="5%", pad=0.1)
#            plt.colorbar(im, cax=cax)
#            
#        fig.tight_layout()
#        return fig
#
#
#    def plot_feature_map(model, layer_id, X, n=256, ax=None, **kwargs):
#        """
#        """
#        import keras.backend as K
#        import matplotlib.pyplot as plt
#        from mpl_toolkits.axes_grid1 import make_axes_locatable
#
#        layer = model.layers[layer_id]
#        
#        try:
#            get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
#            activations = get_activations([X, 0])[0]
#        except:
#            # Ugly catch, a cleaner logic is welcome here.
#            raise Exception("This layer cannot be plotted.")
#
#        activationss=activations.reshape(activations.shape[0], activations.shape[3], activations.shape[1], activations.shape[2])    
#        # For now we only handle feature map with 4 dimensions
#        if activationss.ndim != 4:
#            raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
#                                                                                                activations.ndim))
#            
#        # Set default matplotlib parameters
#        if not 'interpolation' in kwargs.keys():
#            kwargs['interpolation'] = "none"
#
#        if not 'cmap' in kwargs.keys():
#            kwargs['cmap'] = "hot"
#            
#        fig = plt.figure(figsize=(15, 15))
#        
#        # Compute nrows and ncols for images
#        n_mosaic = len(activationss)
#        print('Test n_mosaic:', n_mosaic)
#        nrows = int(np.round(np.sqrt(n_mosaic)))
#        ncols = int(nrows)
#        if (nrows ** 2) < n_mosaic:
#            ncols +=1
#            
#        
#        # Compute nrows and ncols for mosaics
#        if activationss[0].shape[0] < n:
#            n = activationss[0].shape[0]
#        print('Test n:', n)
#        print('test shape = 1',activationss[0].shape[1])   
#        print('test shape = 2',activationss[0].shape[2])    
#        nrows_inside_mosaic = int(np.round(np.sqrt(n)))
#        ncols_inside_mosaic = int(nrows_inside_mosaic)
#
#        if nrows_inside_mosaic ** 2 < n:
#            ncols_inside_mosaic += 1
#
#        for i, feature_map in enumerate(activationss):
#
#            mosaic = make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)
#
#            
#            ax = fig.add_subplot(nrows, ncols, i+1)
#            
#            im = ax.imshow(mosaic, **kwargs)
#            ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
#                                                                                    layer.name,
#                                                                                    layer.__class__.__name__))
#
#            divider = make_axes_locatable(ax)
#            cax = divider.append_axes("right", size="5%", pad=0.1)
#            plt.colorbar(im, cax=cax)
#                
#            fig.tight_layout()
#        return fig
#
#
#    def plot_all_feature_maps(model, X, n=256, ax=None, **kwargs):
#        
#        figs = []
#        
#        for i, layer in enumerate(model.layers):
#            
#            try:
#                fig = plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
#            except:
#                pass
#            else:
#                figs.append(fig)
#                
#        return figs
#    _ = plot_feature_map(model,1, x_test[:3], n=36)