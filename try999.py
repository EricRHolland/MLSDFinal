# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:45:41 2021

@author: EricH
"""

import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

sys.path.append("tl_gan")
sys.path.append("pg_gan")
# import feature_axis
# import tfutil
# import tfutil_cpu

def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return
        
def orthogonalize_one_vector(vector, vector_base):
    """
    tool function, adjust vector so that it is orthogonal to vector_base (i.e., vector - its_projection_on_vector_base )
    :param vector0: 1D array
    :param vector1: 1D array
    :return: adjusted vector1
    """
    return vector - np.dot(vector, vector_base) / np.dot(vector_base, vector_base) * vector_base

def normalize_feature_axis(feature_slope):
    """
    function to normalize the slope of features axis so that they have the same length
    :param feature_slope: array of feature axis, shape = (num_latent_vector_dimension, num_features)
    :return: same shape of input
    """

    feature_direction = feature_slope / np.linalg.norm(feature_slope, ord=2, axis=0, keepdims=True)
    return feature_direction


def disentangle_feature_axis(feature_axis_target, feature_axis_base, yn_base_orthogonalized=False):
    """
    make feature_axis_target orthogonal to feature_axis_base
    :param feature_axis_target: features axes to decorrerelate, shape = (num_dim, num_feature_0)
    :param feature_axis_base: features axes to decorrerelate, shape = (num_dim, num_feature_1))
    :param yn_base_orthogonalized: True/False whether the feature_axis_base is already othogonalized
    :return: feature_axis_decorrelated, shape = shape = (num_dim, num_feature_0)
    """

    # make sure this funciton works to 1D vector
    if len(feature_axis_target.shape) == 0:
        yn_single_vector_in = True
        feature_axis_target = feature_axis_target[:, None]
    else:
        yn_single_vector_in = False

    # if already othogonalized, skip this step
    if yn_base_orthogonalized:
        feature_axis_base_orthononal = orthogonalize_vectors(feature_axis_base)
    else:
        feature_axis_base_orthononal = feature_axis_base

    # orthogonalize every vector
    feature_axis_decorrelated = feature_axis_target + 0
    num_dim, num_feature_0 = feature_axis_target.shape
    num_dim, num_feature_1 = feature_axis_base_orthononal.shape
    for i in range(num_feature_0):
        for j in range(num_feature_1):
            feature_axis_decorrelated[:, i] = orthogonalize_one_vector(feature_axis_decorrelated[:, i],
                                                                       feature_axis_base_orthononal[:, j])

    # make sure this funciton works to 1D vector
    if yn_single_vector_in:
        result = feature_axis_decorrelated[:, 0]
    else:
        result = feature_axis_decorrelated

    return result
def orthogonalize_vectors(vectors):
    """
    tool function, adjust vectors so that they are orthogonal to each other, takes O(num_vector^2) time
    :param vectors: vectors, shape = (num_dimension, num_vector)
    :return: orthorgonal vectors, shape = (num_dimension, num_vector)
    """
    vectors_orthogonal = vectors + 0
    num_dimension, num_vector = vectors.shape
    for i in range(num_vector):
        for j in range(i):
            vectors_orthogonal[:, i] = orthogonalize_one_vector(vectors_orthogonal[:, i], vectors_orthogonal[:, j])
    return vectors_orthogonal


def disentangle_feature_axis_by_idx(feature_axis, idx_base=None, idx_target=None, yn_normalize=True):
    """
    disentangle correlated feature axis, make the features with index idx_target orthogonal to
    those with index idx_target, wrapper of function disentangle_feature_axis()
    :param feature_axis:       all features axis, shape = (num_dim, num_feature)
    :param idx_base:           index of base features (1D numpy array), to which the other features will be orthogonal
    :param idx_target: index of features to disentangle (1D numpy array), which will be disentangled from
                                    base features, default to all remaining features
    :param yn_normalize:       True/False to normalize the results
    :return:                   disentangled features, shape = feature_axis
    """
    (num_dim, num_feature) = feature_axis.shape

    # process default input
    if idx_base is None or len(idx_base) == 0:    # if None or empty, do nothing
        feature_axis_disentangled = feature_axis
    else:                                         # otherwise, disentangle features
        if idx_target is None:                # if None, use all remaining features
            idx_target = np.setdiff1d(np.arange(num_feature), idx_base)

        feature_axis_target = feature_axis[:, idx_target] + 0
        feature_axis_base = feature_axis[:, idx_base] + 0
        feature_axis_base_orthogonalized = orthogonalize_vectors(feature_axis_base)
        feature_axis_target_orthogonalized = disentangle_feature_axis(
            feature_axis_target, feature_axis_base_orthogonalized, yn_base_orthogonalized=True)

        feature_axis_disentangled = feature_axis + 0  # holder of results
        feature_axis_disentangled[:, idx_target] = feature_axis_target_orthogonalized
        feature_axis_disentangled[:, idx_base] = feature_axis_base_orthogonalized

    # normalize output
    if yn_normalize:
        feature_axis_out = normalize_feature_axis(feature_axis_disentangled)
    else:
        feature_axis_out = feature_axis_disentangled
    return feature_axis_out


import sys


sys.path.append("tl_gan")
sys.path.append("pg_gan")


st.title("Streamlit Face-GAN Demo")


# Ensure that load_pg_gan_model is called only once, when the app first loads.
#@st.cache(allow_output_mutation=True)
def load_pg_gan_model():
    """
    Create the tensorflow session.
    """
    # Open a new TensorFlow session.
        # Read in either the GPU or the CPU version of the GAN
    with open(MODEL_FILE_GPU if USE_GPU else MODEL_FILE_CPU, "rb") as f:
        G = pickle.load(f)
    return G


# Ensure that load_tl_gan_model is called only once, when the app first loads.
# @st.cache(hash_funcs=TL_GAN_HASH_FUNCS)
def load_tl_gan_model():
    """
    Load the linear model (matrix) which maps the feature space
    to the GAN's latent space.
    """
    with open(FEATURE_DIRECTION_FILE, "rb") as f:
        feature_direction_name = pickle.load(f)

    # Pick apart the feature_direction_name data structure.
    feature_direction = feature_direction_name["direction"]
    feature_names = feature_direction_name["name"]
    num_feature = feature_direction.shape[1]
    feature_lock_status = np.zeros(num_feature).astype("bool")

    # Rearrange feature directions using Shaobo's library function.
    feature_direction_disentangled = disentangle_feature_axis_by_idx(
        feature_direction, idx_base=np.flatnonzero(feature_lock_status)
    )
    return feature_direction_disentangled, feature_names


def get_random_features(feature_names, seed):
    """
    Return a random dictionary from feature names to feature
    values within the range [40,60] (out of [0,100]).
    """
    np.random.seed(seed)
    features = dict((name, 40 + np.random.randint(0, 21)) for name in feature_names)
    return features


# Hash the TensorFlow session, the pg-GAN model, and the TL-GAN model by id
# to avoid expensive or illegal computations.
# @st.cache(show_spinner=False, hash_funcs=TL_GAN_HASH_FUNCS)
def generate_image(session, pg_gan_model, tl_gan_model, features, feature_names):
    """
    Converts a feature vector into an image.
    """
    # Create rescaled feature vector.
    feature_values = np.array([features[name] for name in feature_names])
    feature_values = (feature_values - 50) / 250
    # Multiply by Shaobo's matrix to get the latent variables.
    latents = np.dot(tl_gan_model, feature_values)
    latents = latents.reshape(1, -1)
    dummies = np.zeros([1] + pg_gan_model.input_shapes[1][1:])
    # Feed the latent vector to the GAN in TensorFlow.
    with session.as_default():
        images = pg_gan_model.run(latents, dummies)
    # Rescale and reorient the GAN's output to make an image.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
        np.uint8
    )  # [-1,1] => [0,255]
    if USE_GPU:
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return images[0]




USE_GPU = False
FEATURE_DIRECTION_FILE = "feature_direction_2018102_044444.pkl"
MODEL_FILE_GPU = "karras2018iclr-celebahq-1024x1024-condensed.pkl"
MODEL_FILE_CPU = "karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl"
EXTERNAL_DEPENDENCIES = {
    "feature_direction_2018102_044444.pkl": {
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/feature_direction_20181002_044444.pkl",
        "size": 164742,
    },
    "karras2018iclr-celebahq-1024x1024-condensed.pkl": {
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024-condensed.pkl",
        "size": 92338293,
    },
    "karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl": {
        "url": "https://streamlit-demo-data.s3-us-west-2.amazonaws.com/facegan/karras2018iclr-celebahq-1024x1024-condensed-cpu.pkl",
        "size": 92340233,
    },
}

# Download all data files if they aren't already in the working directory.
for filename in EXTERNAL_DEPENDENCIES.keys():
    download_file(filename)

# Read in models from the data files.
tl_gan_model, feature_names = load_tl_gan_model()
pg_model = load_pg_gan_model()

st.sidebar.title("Features")
seed = 27834096
# If the user doesn't want to select which features to control, these will be used.
default_control_features = ["Young", "Smiling", "Male"]

if st.sidebar.checkbox("Show advanced options"):
    # Randomly initialize feature values.
    features = get_random_features(feature_names, seed)

    # Some features are badly calibrated and biased. Removing them
    block_list = ["Attractive", "Big_Lips", "Big_Nose", "Pale_Skin"]
    sanitized_features = [
        feature for feature in features if feature not in block_list
    ]

    # Let the user pick which features to control with sliders.
    control_features = st.sidebar.multiselect(
        "Control which features?",
        sorted(sanitized_features),
        default_control_features,
    )
else:
    features = get_random_features(feature_names, seed)
    # Don't let the user pick feature values to control.
    control_features = default_control_features

# Insert user-controlled values from sliders into the feature vector.
for feature in control_features:
    features[feature] = st.sidebar.slider(feature, 0, 100, 50, 5)

st.sidebar.title("Note")
st.sidebar.write(
    """Playing with the sliders, you _will_ find **biases** that exist in this
    model.
    """
)
st.sidebar.write(
    """For example, moving the `Smiling` slider can turn a face from masculine to
    feminine or from lighter skin to darker. 
    """
)
st.sidebar.write(
    """Apps like these that allow you to visually inspect model inputs help you
    find these biases so you can address them in your model _before_ it's put into
    production.
    """
)
st.sidebar.caption(f"Streamlit version `{st.__version__}`")

# Generate a new image from this feature vector (or retrieve it from the cache).
# with session.as_default():
#     image_out = generate_image(
#         session, pg_gan_model, tl_gan_model, features, feature_names
#     )

# st.image(image_out, use_column_width=True)



# These are handles to two visual elements to animate.
weights_warning, progress_bar = None, None
try:
    weights_warning = st.warning("Downloading %s..." % file_path)
    progress_bar = st.progress(0)
    with open(file_path, "wb") as output_file:
        with urllib.request.urlopen(
            EXTERNAL_DEPENDENCIES[file_path]["url"]
        ) as response:
            length = int(response.info()["Content-Length"])
            counter = 0.0
            MEGABYTES = 2.0 ** 20.0
            while True:
                data = response.read(8192)
                if not data:
                    break
                counter += len(data)
                output_file.write(data)

                # We perform animation by overwriting the elements.
                weights_warning.warning(
                    "Downloading %s... (%6.2f/%6.2f MB)"
                    % (file_path, counter / MEGABYTES, length / MEGABYTES)
                )
                progress_bar.progress(min(counter / length, 1.0))

# Finally, we remove these visual elements by calling .empty().
finally:
    if weights_warning is not None:
        weights_warning.empty()
    if progress_bar is not None:
        progress_bar.empty()


# Make note without using cv2.read or cv2. imwrite because those crash in streamlit. How to get this working again?
# imwrite function crashes because it's invalid, but some other functions crash because of the input
#maybe have to pass the image into a different type of numpy array
# try using some sampple code, if it still doesnt work resort to previous working version from main file
    

# img = cv2.imread(image_test)
# faces = face_cascade.detectMultiScale(img, 1.1, 4)
# for(x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y),
#                   (x+w, y+h), (25,25,255),thickness = 4)
# a = cv2.imwrite('face_detected_image.png', img) 

# print('successfully saved.')


