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



### Changes in version 17/03/2018

- train.py

  - Back to plotting every 200 mini-batches
  - now using a different architecture in bottleneck (doubling channels and dividing by two the size at every conv, same at deconv)


- predict.py

  - using fixed batch of 64 images for testing, those image IDs can be found in dataset_lungs/test_64
  - now returning average PSNR over patches (best result: ~31) and over images (best result: ~37) for every minibatch


- psnr.py
  
  - computes psnr for an image


- plotter.py

  - the plotting function is now separated from the main.py


- train512.py and model512.py

  - experiment ready adding more layers to fit an input image of 512 (center cropped from original 1024) and patch size 128, with both initial channels = 128 for Discriminator and Generator, being doubled at every conv step




### Changes in version 19/03/2018

- train.py

  - all hyperparameters can be set by command line arguments
  - all outputs are now well-organized in folders (check PATH_* variables in the code for directories name).
  - the main directory containing results includes description of 4 main architecture parameters (imageSize, patchSize, nef, ndf), plus a name for that experiment (and folder) can be specified with --name
  - prediction on testset are now executed at the end of every epoch and saved in a subdirectory.
  - PSNR values for every epoch on test step are now saved in a .txt file in the test subdirectory.
  - randomCrop experiment allow to run that particular experiment, additionally setting parameters as padding, number of crops per image (dataset size will be n_crops times n_images), size of central area from which get randomCrops.
  - testing the recunstruction of the whole image adding back the randomCrops is now added to the test phase, but currently not running as expected (some bug to be fixed)
  - all unused code, parameters and comments are removed

- model.py

  - both context-encoder and discriminator are now created automatically depending on imageSize, patchSize, nef, ndf using the architecture discussed
  - added names and description for every layer to be displayed when the model is created at runtime