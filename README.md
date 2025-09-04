# Controllable latent diffusion model to evaluate the performance of cardiac segmentation methods



## Prerequisites
- Python 3.10+


## Installation
- Clone the repository and enter it:
    - git clone <https://github.com/creatis-myriad/cLDM_project>
    - `cd cLDM_poject/bin/`


## Create the Conda environment
- Create from the provided environment file:
    - conda env create -f environment.yml
- Activate it:
    - conda activate cLDM_env


## Data preparation
- Data from MYOSAIQ challenge were put in one folder and divided by patients.
- The D8 subset was not used during training.
- In the code, `D_metrics` is a dictionnary with specific informations (like the metrics) obtained from the MYOSAIQ database.


## Configuration files
- Configurations files that were used to run the experiments can be found in the folder `nn_lib/config/`. It is configurated as follow:
    - `Config_VanillaVAE.yaml` is the main config file that will use other configuration files to run the VAE model.
    - The folder `config/model` contains the configuration files with the parameters for the each model.
    - The folder `config/architecture` contains the configuration files to select specific architecture depending and the model you want to run.
    - The folders `config/dataset` and `config/datamodule` contain the configuration files to create the dataloaders for the training of the model.
- When running a main configuration, the sub-configuration files used should have the same name.


## Models used for the experiments
- For strategy 1, the `cLDM` model is used. The conditioning (using cross-attention) was done with:
    - A vector of scalars derived from the segmentations (strategy 1.1).
    - The latent representation from `VanillaVAE` of the images (strategy 1.2). 
    - The latent representation from `ARVAE` of the images and regularized with clinical attributes (strategy 1.3)
- For strategy 2, the `ControlNet` model is used using the `LDM` model.
- For strategy 3, the `cLDM_concat` model is used using 2D representation of segmentation masks obtained from the `AE_KL` model.


## How to run
Below are the command line to run the models:
    - **VanillaVAE**
        ```    
        # For Images
        python nn_models/bin/train_VanillaVAE.py \
        +config_name=JobConfig_VAE.yaml \
        \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=500 \
        model.net.shape_data=[1,128,128] \
        model.net.lat_dims=8 \
        model.net.alpha=5 \
        model.net.beta=8e-3 \
        ```
    - **test** 




## Figure
TODO
<!-- 
File fig_ ... was the one use to create fig 2 and 3 in the article.
 -->












