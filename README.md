### Copyright (C) Microsoft Corporation.  


Neural Networks (NN) are a machine learning technique that has been used in academia and research for many decades. The main reason why they 
were not widely used in the industry until recently was that they were computationally intensive and extremely hard to train over many layers 
(deep networks). However, NNs have become very popular recently precisely because this big challenge has been solved by using predefined NN 
architectures that are easy to replicate, although they remain computationally intensive.  

One of the main focuses of such predefined multi layered (deep) NN architectures is image classification using convolutional NNs (CNN), a specific 
type of NN that uses shared NN layers weights to learn hierarchical shift invariant representations of images. This approach industrializes image 
preprocessing, by automating the images’ encoding into features (a.k.a. feature engineering) process, which traditionally required both computer 
science (image processing) knowledge and advanced subject matter expertise. Moreover, the main hurdle of training multi-layer CNNs can be addressed 
through “transfer learning”, a technique that uses CNNs pre-trained on generic image datasets as initial starting points for training domain 
specific classifiers. While the weights of the pre-trained CNNs can be further refined by continuing training on domain specific data, the most 
straightforward example of transfer learning is to “freeze” the network and use it to compute image features that are then fed into a regular 
machine learning algorithm to build a prediction model. This allows for virtually out-of-the-box building of powerful baseline deep learning 
models for virtually any domain, from medical images like computed tomography scans and X-ray images to industrial optical images or satellite 
imagery.  

This approach can be further generalized to non-image datasets like IoT time series, where CNN are usually less used presumably because
of the perceived lack the 2D structure that can be extracted though convolutional layers. However, traditional deep-learning methods for time 
based recordings that use the time component directly through recurrent neural network (RNN) methods like long short-term memory (LSTM) algorithm, 
can in fact be directly compared against CNNs by emphasizing the multichannel 1D images properties of IoT data. 

A significant by-product of CNN based deep learning methods is that by using specific final trainable layers architectures like global average 
pooling (GAP) one can map the location of image areas that contribute to specific predictions. Individual prediction activation maps like Class 
Activation Mapping (CAM) images allow one to understand what the model learns and thus explain a prediction output. This information is extracted 
by the trained model from new images, which is extremely powerful considering that the class specific location information is also not provided 
at training stage. Besides model explainability, CAM images can be used for model improvement, through guided training by removing (or relabeling) 
in training data the areas that should not be the focus for prediction.  

Here we use GPU enabled Deep Learning Virtual Machines (DLVM) available on Microsoft Azure AI platform to show how data scientists can leverage 
open-source deep learning frameworks like Keras to build end to end intelligent signal classification solutions powered by deep learning models.  


Instructions:
Install Azure Machine Learning Workbench AMLWB (https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation).  

Open an amlwb cli and follow this [guide](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-2#execute-scripts-in-the-azure-machine-learning-cli-window) and this Azure ML o16n [cheat sheet](https://gist.github.com/georgeAccnt-GH/028c376f3139a445ba3d19705418da5f) to create an AMLWB worspace, run ML experiments, and deploy models: 

1. Set up your environment:   
  REM login by using the aka.ms/devicelogin site      
  az login      
        
  REM lists all Azure subscriptions you have access to (# make sure the right subscription is selected (column isDefault))      
  az account list -o table      
        
  REM sets the current Azure subscription to the one you want to use      
  az account set -s <subscriptionId>      
        
  REM verifies that your current subscription is set correctly      
  az account show      
     
  REM  Create an experimentation account and and azure ml workspace using the portal   
     
  REM  Use the AMLWB to create a new project   
     
  REM Copy \Code\ structure and files (.ipynb and .py files) in the new experiment folder   
   
          
          
2. Create compute contexts on remote VMs:        
        
  2.1  Using Azure portal:     
  - Deploy a linux VM (e.g. a linux [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro))      
     For best results, use a a deep learning linux VM (https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning).   
     You may need a [GPU VM](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) for training, and a second CPU VM for operationalization testing.