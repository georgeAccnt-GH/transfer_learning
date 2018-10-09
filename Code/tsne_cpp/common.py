import sys
import os
from keras.layers import Dense
from keras.models import Model
from keras_contrib.applications.densenet import DenseNetImageNet121
import keras_contrib
from keras.applications.resnet50 import ResNet50
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import tempfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import t_sne_bhcuda.t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda


def apply_tsne_to_multiple_features_cpp(output_dir,
                                        model_name_list,
                                        model_layer_list,
                                        interal_dir=None,
                                        perplexity=30.0,
                                        theta=0.5,
                                        learning_rate=200.0,
                                        iterations=1000,
                                        gpu_mem=0.8,
                                        batch_size=0,
                                        saved_data_file_appendix='',
                                        recompute=False):
    # FIXME: refactor this
    for crt_model_name, crt_model_layer in zip(model_name_list, model_layer_list):
        print('tsne processing for model ' + crt_model_name.__name__)
        base_filename = 'features_' + crt_model_name.__name__ + \
            saved_data_file_appendix + crt_model_layer
        tsne_features_filename = os.path.join(output_dir,
                                              base_filename + '_tsne_cpp.npy')
        if os.path.isfile(tsne_features_filename) and recompute is False:
            print("tsne features found! Skipping...")
        else:
            features_path = os.path.join(output_dir, base_filename + '.npy')
            print("Loading image features from {}".format(features_path))
            #original_features = np.load(features_path)
            original_features = np.random.rand(20000, 2048)
            print(original_features.shape)
            print(type(original_features[0, 0]))
            original_features = original_features.reshape(
                original_features.shape[0], -1)
            print(original_features.shape)
            if interal_dir is None:
                interal_dir = os.path.join(tempfile.gettempdir(), 'tsne_cpp')
            print("Computing tsne features")
            images_tsne = tsne_bhcuda.t_sne(samples=original_features,
                                            use_scikit=False,
                                            files_dir=interal_dir,
                                            no_dims=2,
                                            perplexity=perplexity,
                                            eta=learning_rate,
                                            theta=theta,
                                            iterations=iterations,
                                            gpu_mem=gpu_mem,
                                            seed=batch_size,
                                            randseed=-1,
                                            verbose=2)

            print("tsne features shape: {}".format(images_tsne.shape))
            np.save(tsne_features_filename, images_tsne)
            print("Features saved to {}".format(tsne_features_filename))


def visualize_tsne_features(output_dir,
                            model_name_list,
                            model_layer_list,
                            y,
                            saved_data_file_appendix=''):
    # FIXME: refactor this
    for crt_model_name, crt_model_layer in zip(model_name_list, model_layer_list):
        base_filename = 'features_' + crt_model_name.__name__ + \
            saved_data_file_appendix + crt_model_layer
        tsne_features_filename = os.path.join(output_dir,
                                              base_filename + '_tsne_cpp.npy')
        tsne_features = np.load(tsne_features_filename)
        plt.figure(figsize=(8, 8))
        plt.scatter(x=tsne_features[:, 0], y=tsne_features[:, 1],
                    marker=".", c=y, cmap=plt.cm.get_cmap('bwr'))
        plt.show()
