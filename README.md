# Context-Encoder PyTorch Implementation

based on : [context-encoder-pytorch](https://github.com/BoyuanJiang/context_encoder_pytorch)

[Davide Belli](https://github.com/davide-belli)

### First version 01/03/2018

- train.py 
  - It is used to train the model, it accepts some parameters for execution from command line, although I set my default configuration to have it running without other params.
  - Outputs:
    - ./model directory with the Generative and Discrimative model files as pytorch objects, to be used for predictions (the models are updated and saved every epoch)
    - ./result/lungs directory containing a sample resulting minibatch for each epoch (on train data), updated very 100 steps in the same epoch
    - plot.png and plot_p.png plotting P(real), P(Fake), Adv_loss averaged over 200 minibatches
    - measures.pickle containing the data plotted in plot.png

  - In my configuration I use by default CUDA for training.
  - Also, I assume my dataset is in the directory:
    - dataset_lungs/train/ containing SUBDIRECTORIES with all the images to be trained on
    - dataset_lungs/test/ containing SUBDIRECTORIES with all images for testing with predict.py


- predict.py
   - to use a saved model for predictions on the test set. plots reconstructed and real images in: ./predict/lungs
   - by default, the models (D and G) must be in the ./model directory, as outputed from train.py


- image_selector.ipynb
  - Jupyter python script to generate my dataset from the downloaded one given a list of images in no_findings.txt



### Changes in version 10/03/2018

- train.py

  - Dataset transformed to Grayscale, nc=1 channels by default (previous was 3)
  - More powerful Discriminator doubling channels ndf=128 (previous was 64)
  - Context Encoder is only trained every two Discriminator updates (2 minibatches). Set variable N_UPDATE_GEN to change this.
  - CONTINUE_TRAINING to continue training a model saved in ./model
  - Added option to Normalize Data (NOT used at the moment, commented code)
  - Fixed and improved plots (Advs loss was wrong, added labels and different plots). Plotting loss/values of every minibatch (previously was every 200 minibatches)


- predict.py

  - Input model has nc=1 channel (added Grayscale transformation in train.py)
  - Input model has ndf=128 channels (improved Discriminator from 64 in older versions)
  - Reconstruction on the test set now output the same minibatches to make comparison easier between different models. The number of output minibatches is defined in LIMIT_SAMPLES
