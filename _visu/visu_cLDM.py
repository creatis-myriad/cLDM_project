import os
import pickle
import sys
import torch

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../') 

from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from nn_lib.common.funPath import GetPaths
from nn_lib.models.ARVAE import ARVAE
from nn_lib.models.AE_KL import AE_KL
from nn_lib.models.cLDM import cLDM
from nn_lib.models.VanillaVAE import VanillaVAE
from nn_lib.utils import format



def MSE(A, B) :
    return ((A - B)**2).mean()



def main() :
    print("Loading parameters ...")
    path_ = sys.argv[-1] # Dir of the model
    path_model = GetPaths(path_, ["last.ckpt"])["last.ckpt"][-1]
    D_model_DDPM = torch.load(path_model)
    cfg = D_model_DDPM["hyper_parameters"]["config"]
    seed_everything(cfg.model.seed, workers=False)
    device = torch.device(cfg.model.device)


    print(f"Instantiating Datamodule <{cfg.datamodule._target_}>")
    path_compress_data = GetPaths(cfg.model.path_model_autoencoder, ["output_for_ldm"])["output_for_ldm"][0]
    if cfg.model.path_model_cond :
        path_LS_cond_data  = GetPaths(cfg.model.path_model_cond, ["embedding"])["embedding"][0]
    else : path_LS_cond_data = None

    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg=cfg
    Datamodule_.path_compress_data = path_compress_data
    Datamodule_.path_cond_data = path_LS_cond_data
    Datamodule_.setup()
    full_data = Datamodule_.predict_dataloader()
    shape_data = Datamodule_.train[0]["input_mu"].shape
    nb_data = len(Datamodule_.test)
    if nb_data > 500: nb_data = 500    


    L_pat_name = []
    L_seq_name = []
    L_input_mod1 = []
    L_input_mod2 = []
    for sample in full_data :
        L_input_mod1.append(sample["input_mod1"])
        L_input_mod2.append(sample["input_mod2"])
        L_pat_name.append(sample["pat_name"])
        L_seq_name.append(sample["seq_name"])
    L_input_mod1 = np.concatenate(L_input_mod1, axis=0)
    L_input_mod2 = np.concatenate(L_input_mod2, axis=0)
    L_pat_name = np.concatenate(L_pat_name, axis=0)
    L_seq_name = np.concatenate(L_seq_name, axis=0)


    print("Instantiating model <AutoencoderKL>")
    path_AEKL = cfg.model.path_model_autoencoder
    path_model_AEKL = GetPaths(path_AEKL, [".ckpt"])[".ckpt"][-1]
    D_model_AEKL = torch.load(path_model_AEKL)
    cfg_AEKL = D_model_AEKL["hyper_parameters"]["config"]

    autoencoderKL = instantiate(cfg_AEKL.architecture.autoencoders.autoencoderKL)
    patchDiscriminator = instantiate(cfg_AEKL.architecture.autoencoders.patchDiscriminator)

    AE_KL_model = AE_KL(
        config=cfg_AEKL,
        autoencoderKL=autoencoderKL,
        patchDiscriminator=patchDiscriminator,
        learning_rate_generator=cfg_AEKL.model.net.lr_g,
        learning_rate_discriminator=cfg_AEKL.model.net.lr_d,
    ).to(device)

    AE_KL_model.load_state_dict(
        D_model_AEKL["state_dict"], 
    )
    for param in AE_KL_model.parameters(): param.requires_grad = False


    if cfg.model.path_model_cond :
        path_conditional_model = cfg.model.path_model_cond
        path_model_conditional_model = GetPaths(path_conditional_model, [".ckpt"])[".ckpt"][-1]
        D_model_conditional_model = torch.load(path_model_conditional_model)
        cfg_cond_model = D_model_conditional_model["hyper_parameters"]["config"]
        lat_dims = cfg_cond_model.model.net.lat_dims
        
        if "training_VAE/" in path_model_conditional_model :
            print("Instantiating model <VanillaVAE>")

            enc = instantiate(cfg_cond_model.architecture.encoders).to(device)
            dec = instantiate(cfg_cond_model.architecture.decoders).to(device)

            cond_model = VanillaVAE(
                config=cfg_cond_model,
                encoder=enc,
                decoder=dec,
                lat_dim=lat_dims,
                learning_rate=cfg_cond_model.model.net.lr,
                alpha=cfg_cond_model.model.net.alpha,
                beta=cfg_cond_model.model.net.beta,
            ).to(device)

        if "training_ARVAE/" in path_model_conditional_model :
            print("Instantiating model <ARVAE>")

            enc = instantiate(cfg_cond_model.architecture.encoders).to(device)
            dec = instantiate(cfg_cond_model.architecture.decoders).to(device)

            cond_model = ARVAE(
                config=cfg_cond_model,
                encoder=enc,
                decoder=dec,
                lat_dim=lat_dims,
                learning_rate=cfg_cond_model.model.net.lr,
                alpha=cfg_cond_model.model.net.alpha,
                beta=cfg_cond_model.model.net.beta,
                gamma=cfg_cond_model.model.net.gamma,
                delta=cfg_cond_model.model.net.delta,
            ).to(device)

        cond_model.load_state_dict(
            D_model_conditional_model["state_dict"], 
        )
        for param in cond_model.parameters(): param.requires_grad = False
    else : cond_model = None


    print("Instantiating model <cLDM>")
    cLDM_model = cLDM(
        config=cfg,
        model=instantiate(cfg.architecture.unets), 
        autoencoder_model=AE_KL_model,
        conditional_model=cond_model,
        learning_rate=cfg.model.net.lr,
        num_timesteps=cfg.model.net.num_timesteps, 
        scheduler_type=cfg.model.net.scheduler,
        scale_factor=cfg.model.net.scale_factor,
    ).to(device)

    cLDM_model.load_state_dict(
        D_model_DDPM["state_dict"], 
    )


    print()
    print()
    print("Figures for validating conditional Latent Diffusion Model\n")
    name_nn = sys.argv[-1].split("/")[-1]
    output_path = "/home/deleat/Documents/RomainD/Working_space/Figures_Resultats/Results_LLDM_project/validation_conditionnement_cLDM/"
    output_path = os.path.join(output_path, name_nn)

    L_cond = []
    for elem in Datamodule_.train :
        L_cond.append(elem["input_cond"])
    for elem in Datamodule_.val :
        L_cond.append(elem["input_cond"])
    for elem in Datamodule_.test :
        L_cond.append(elem["input_cond"])
    L_cond = np.array(L_cond).squeeze()


    no_specific_keys = True
    with open(cfg.processing.path_metrics, 'rb') as fileMetrics:
        D_metrics = pickle.load(fileMetrics)
    if "keys_cond_data" in cfg_cond_model.model.net: 
        keys_metrics = cfg_cond_model.model.net.keys_cond_data
        no_specific_keys = False
    else: keys_metrics = D_metrics.keys()

    idx_copy = -1
    D_metrics_dataset = {}
    for pat_name in L_pat_name :
        idx = D_metrics["pat_name"].index(pat_name)
        if idx != idx_copy : cpt = 0; idx_copy = idx
        for key in keys_metrics :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                if key not in D_metrics_dataset : D_metrics_dataset[key] = []
                D_metrics_dataset[key].append(D_metrics[key][idx][cpt])
            else:
                if key not in D_metrics_dataset : D_metrics_dataset[key] = []
                D_metrics_dataset[key].append(D_metrics[key][idx])
        cpt+=1
    if no_specific_keys : keys_metrics = D_metrics_dataset.keys()


    if cfg_AEKL.dataset.work_on == "dcm.pkl" : 
        L_input = np.copy(L_input_mod1); vmax = 1
    else : 
        L_input = np.copy(L_input_mod2); vmax = 2
    


    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)

    print("generating Batch from trained latent vector ...")
    name_folder = f"Batch from 1 latent vector (from train)"
    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
            os.makedirs(output_folder)

    idx = 8
    latent_vec = L_cond[idx:idx+1]
    latent_vec = np.concatenate([latent_vec for i in range (35)], axis=0)
    if len(latent_vec.shape) == 1 : latent_vec = latent_vec[...,None]
    latent_vec = torch.tensor(latent_vec).to(device)
    variation_of_latent = cLDM_model.generate(latent_vec, shape_data)
    variation_of_latent = format.torch_to_numpy(variation_of_latent)
    variation_of_latent = np.concatenate([variation_of_latent, L_input[idx][None,...]], axis=0)

    print("   closest images ...")
    L_closest_img = []
    for gen_imgs in variation_of_latent :
        L_comparaison = [MSE(gen_imgs, input_img.squeeze()) for input_img in L_input]
        idx = np.argmin(L_comparaison)
        L_closest_img.append(L_input[idx])
    L_closest_img = np.array(L_closest_img)

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Closest imgs of variations of a latent vector (train)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        plt.imshow(L_closest_img[i,0,...], cmap="gray", vmin=0, vmax=vmax)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_closest_train_imgs.png"))

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Generation of variations of a latent vector (train)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        plt.imshow(variation_of_latent[i,0,...], cmap="gray", vmin=0, vmax=vmax)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_train_imgs.png"))


    print("generating Batch from tested latent vector ...")
    idx = 3500
    latent_vec = L_cond[idx:idx+1]
    latent_vec = np.concatenate([latent_vec for i in range (35)], axis=0)
    if len(latent_vec.shape) == 1 : latent_vec = latent_vec[...,None]
    latent_vec = torch.tensor(latent_vec).to(device)
    variation_of_latent = cLDM_model.generate(latent_vec, shape_data)
    variation_of_latent = format.torch_to_numpy(variation_of_latent)
    variation_of_latent = np.concatenate([variation_of_latent, L_input[idx][None,...]], axis=0)

    print("   closest images ...")
    L_closest_img = []
    for gen_imgs in variation_of_latent :
        L_comparaison = [MSE(gen_imgs, input_img.squeeze()) for input_img in L_input]
        idx = np.argmin(L_comparaison)
        L_closest_img.append(L_input[idx])
    L_closest_img = np.array(L_closest_img)

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Closest imgs of variations of a latent vector (test)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        plt.imshow(L_closest_img[i,0,...], cmap="gray", vmin=0, vmax=vmax)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_closest_test_imgs.png"))

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Generation of variations of a latent vector (test)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        plt.imshow(variation_of_latent[i,0,...], cmap="gray", vmin=0, vmax=vmax)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_test_imgs.png"))



if __name__ == "__main__" :
    main()    










