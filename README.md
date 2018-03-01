## Context-Encoder PyTorch Implementation

based on : https://github.com/BoyuanJiang/context_encoder_pytorch
Davide Belli, 01/03/2018



--- train.py 
is used to train the model, it accepts some parameters for execution from command line, although I set my default configuration to have it running without other params.
Outputs:
- ./model directory with the Generative and Discrimative model files as pytorch objects, to be used for predictions (the models are updated and saved every epoch)
- ./result/lungs directory containing a sample resulting minibatch for each epoch (on train data), updated very 100 steps in the same epoch
- plot.png and plot_p.png plotting P(real), P(Fake), Adv_loss averaged over 200 minibatches
- measures.pickle containing the data plotted in plot.png

In my configuration I use by default CUDA for training.
Also, I assume my dataset is in the directory:
dataset_lungs/train/ containing SUBDIRECTORIES with all the images to be trained on
dataset_lungs/test/ containing SUBDIRECTORIES with all images for testing with predict.py




--- predict.py
 - to use a saved model for predictions on the test set. plots reconstructed and real images in: ./predict/lungs
 - by default, the models (D and G) must be in the ./model directory, as outputed from train.py
