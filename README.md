# Controllable latent diffusion model to evaluate the performance of cardiac segmentation methods


![Pipeline overview](./figures/fig1_pipeline.png "Figure 1.")



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


## Figures from the paper
- Figure 2 and 3 where obtained using the file `fig_originalSeg_vs_generatedSeg.py`. It needs the original segmentation as well as the segmentation derived from the nnU-Net model with synthetic images serving as the input.
- To get **Figure 2**, we have chosen an arbitrary mask to illustrate our pipeline.

![Pipeline overview](./figures/Dice_segGen_v2.png "Figure 2.")

- To get **Figure 3**, we have selected specific masks with relevant characteristics. Therefore, synthetic images were generated and conditioned with those masks, as illustrated in the figure. For the final row, a manual rotation of 90°, 180° and 270° were applied to the mask.

![Pipeline overview](./figures/qualitative_result_v4.png "Figure 3.")


## How to run
Below are the command lines to run the models:
- **VanillaVAE**

    ```bash
    python train_VanillaVAE.py \
        +config_name=Config_VanillaVAE.yaml \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=500 \
        model.net.shape_data=[1,128,128] \
        model.net.lat_dims=8 \
        model.net.alpha=5 \
        model.net.beta=8e-3
    ```

- **ARVAE**

    ```bash
    python train_ARVAE.py \
        +config_name=Config_ARVAE.yaml \
        \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=500 \
        model.net.shape_data=[1,128,128] \
        model.net.lat_dims=8 \
        model.net.alpha=5 \
        model.net.beta=8e-3 \
        model.net.gamma=3 \
        \
        +model.net.keys_cond_data=["z_vals","transmurality","endo_surface_length","infarct_size_2D"] \
    ```

- **AE_KL**

    ```bash
    python train_AE_KL.py \
        +config_name=ConfigAE_KL.yaml \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=1000 \
        model.net.lat_dims=1 \
    ```

- **cLDM**

    ```bash
    # Conditioning with Scalars
    python train_cLDM.py \
        +config_name=Config_cLDM.yaml \
        \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=5100 \
        model.path_model_cond=null \
        \
        processing=processing_CompressLgeSegCond_Scalars \
        dataset=CompressLgeSegCond_Scalars_Dataset \
        datamodule=CompressLgeSegCond_Scalars_Datamodule \
        datamodule.keys_cond_data=["z_vals","transmurality","endo_surface_length","infarct_size_2D"] \
        \
        architecture/unets=unet_cLDM_light \

    ```

    ```bash
    # Conditioning with latent representation from VAE 
    python train_cLDM.py \
        +config_name=Config_cLDM.yaml \
        \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=5100 \
        model.path_model_cond="/home/deleat/Documents/RomainD/Working_space/NN_models/training_Pytorch/training_VAE/training_LgeMyosaiq_v2/2025-01-06 10:13:45_106e_img_base" \
        \
        architecture/unets=unet_cLDM_light \

    ```

    ```bash
    # Conditioning with latent representation from ARVAE
    python nn_models/bin/train_cLDM.py \
        +config_name=Config_cLDM.yaml \
        \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=5100 \
        model.path_model_cond="/home/deleat/Documents/RomainD/Working_space/NN_models/training_Pytorch/training_ARVAE/training_LgeMyosaiq_v2/2025-01-06 14:23:29_72e_img_base" \
        \
        architecture/unets=unet_cLDM_light \
    ```

- **LDM**

    ```bash
    python train_LDM.py \
        +config_name=Config_LDM.yaml \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=5100 \
        architecture/unets=unet_LDM_light \
    ```

- **ControlNet**

    ```bash
    python train_ControlNet.py \
        +config_name=Config_ControlNet.yaml \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=5100 \
    ```

- **cLDM_concat**

    ```bash
    python train_cLDM_concat.py \
        +config_name=Config_cLDM_concat.yaml \
        model.train_params.num_workers=24 \
        model.train_params.batch_size=32 \
        model.train_params.max_epoch=5100 \
        \
        architecture/unets=unet_cLDM_concat_light \
    ```








