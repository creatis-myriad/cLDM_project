import copy
import hydra
import os
import pickle
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../') 

from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from nn_lib.common.funPath import GetPaths
from nn_lib.models.AE_KL import AE_KL
from nn_lib.models.cLDM_concat import cLDM_concat
from nn_lib.utils import format



def MSE(A, B) :
    return ((A - B)**2).mean()



def main() :
    print("Loading parameters ...")
    path_ = sys.argv[-1] # Dir of the model
    path_model = GetPaths(path_, ["last.ckpt"])["last.ckpt"][-1]
    D_model_ = torch.load(path_model)
    cfg = D_model_["hyper_parameters"]["config"]
    seed_everything(cfg.model.seed, workers=False)
    device = torch.device(cfg.model.device)


    print(f"Instantiating Datamodule <{cfg.datamodule._target_}>")
    path_compress_data = GetPaths(cfg.model.path_model_autoencoder, ["output_for_ldm"])["output_for_ldm"][0]
    path_compress_segm = GetPaths(cfg.model.path_model_cond, ["output_for_ldm"])["output_for_ldm"][0]

    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg = cfg
    Datamodule_.path_compress_data = path_compress_data
    Datamodule_.path_compress_segm = path_compress_segm
    Datamodule_.setup()
    full_data = Datamodule_.predict_dataloader()
    shape_data = Datamodule_.train[0]["input_mu"].shape


    keys_metrics =  [
        "z_vals",
        "transmurality",
        "endo_surface_length",
        "infarct_size_2D"
    ]

    L_pat_name = []
    L_seq_name = []
    L_input_mod1 = []
    L_input_mod2 = []
    L_input_metrics = []
    L_input_segm_mu = []
    for sample in full_data :
        L_metrics  = []
        for key in keys_metrics:
            L_metrics.append(sample["metrics"][key])
        L_metrics = np.array(L_metrics).T
    
        L_input_mod1.append(sample["input_mod1"])
        L_input_mod2.append(sample["input_mod2"])
        L_input_segm_mu.append(sample["segm_mu"])
        L_pat_name.append(sample["pat_name"])
        L_seq_name.append(sample["seq_name"])
        L_input_metrics.append(L_metrics)
    L_input_mod1 = np.concatenate(L_input_mod1, axis=0)
    L_input_mod2 = np.concatenate(L_input_mod2, axis=0)
    L_pat_name = np.concatenate(L_pat_name, axis=0)
    L_seq_name = np.concatenate(L_seq_name, axis=0)
    L_input_segm_mu = np.concatenate(L_input_segm_mu, axis=0)
    L_input_metrics = np.concatenate(L_input_metrics, axis=0)


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


    print("Instantiating model <AutoencoderKL_seg>")
    path_AEKL_seg = cfg.model.path_model_cond
    path_model_AEKL_seg = GetPaths(path_AEKL_seg, [".ckpt"])[".ckpt"][-1]
    D_model_AEKL_seg = torch.load(path_model_AEKL_seg)
    cfg_AEKL_seg = D_model_AEKL_seg["hyper_parameters"]["config"]

    autoencoderKL = instantiate(cfg_AEKL_seg.architecture.autoencoders.autoencoderKL)
    patchDiscriminator = instantiate(cfg_AEKL_seg.architecture.autoencoders.patchDiscriminator)

    AE_KL_model_seg = AE_KL(
        config=cfg_AEKL_seg,
        autoencoderKL=autoencoderKL,
        patchDiscriminator=patchDiscriminator,
        learning_rate_generator=cfg_AEKL_seg.model.net.lr_g,
        learning_rate_discriminator=cfg_AEKL_seg.model.net.lr_d,
    ).to(device)

    AE_KL_model_seg.load_state_dict(
        D_model_AEKL_seg["state_dict"], 
    )
    for param in AE_KL_model_seg.parameters(): param.requires_grad = False


    print("Instantiating model <cLDM>")
    cLDM_concat_model = cLDM_concat(
        config=cfg,
        model=instantiate(cfg.architecture.unets), 
        autoencoder_model=AE_KL_model,
        conditional_model=AE_KL_model_seg,
        learning_rate=cfg.model.net.lr,
        num_timesteps=cfg.model.net.num_timesteps, 
        scheduler_type=cfg.model.net.scheduler,
        scale_factor=cfg.model.net.scale_factor,
    ).to(device)

    cLDM_concat_model.load_state_dict(
        D_model_["state_dict"], 
    )


    print()
    print()
    print("Figures for validating conditional Latent Diffusion Model\n")
    name_nn = sys.argv[-1].split("/")[-1]
    output_path = "/home/deleat/Documents/RomainD/Working_space/Figures_Resultats/Results_LLDM_project/validation_conditionnement_cLDM_concat/"
    output_path = os.path.join(output_path, name_nn)


    print("generating Batch from trained latent vector ...")
    name_folder = f"Batch from 1 latent vector (from train)"
    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
            os.makedirs(output_folder)

    idx = 8
    latent_vec = L_input_segm_mu[idx:idx+1]
    latent_vec = np.concatenate([latent_vec for i in range (35)], axis=0)
    if len(latent_vec.shape) == 1 : latent_vec = latent_vec[...,None]
    latent_vec = torch.tensor(latent_vec).to(device)
    variation_of_latent = cLDM_concat_model.generate(latent_vec, shape_data)
    variation_of_latent = format.torch_to_numpy(variation_of_latent)
    variation_of_latent = np.concatenate([variation_of_latent, L_input_mod1[idx][None,...]], axis=0)

    print("   closest images ...")
    cpt = 0
    L_closest_img = []
    nb_ = len(variation_of_latent)
    for gen_imgs in variation_of_latent :
        L_comparaison = [MSE(gen_imgs, input_img.squeeze()) for input_img in L_input_mod1]
        idx = np.argmin(L_comparaison)
        if cpt == nb_ -1 :
            L_closest_img.append(L_input_mod2[idx])
        else:
            L_closest_img.append(L_input_mod1[idx])
        cpt += 1
    L_closest_img = np.array(L_closest_img)

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Closest imgs of variations of a latent vector (train)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        if i < 35 :
            plt.imshow(L_closest_img[i,0,...], cmap="gray", vmin=0, vmax=1)
        else :
            plt.imshow(L_closest_img[i,0,...], cmap="gray", vmin=0, vmax=2)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_closest_train_imgs.png"))

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Generation of variations of a latent vector (train)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        plt.imshow(variation_of_latent[i,0,...], cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_train_imgs.png"))


    print("generating Batch from trained latent vector ...")
    name_folder = f"Batch from 1 latent vector (from test)"
    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
            os.makedirs(output_folder)

    idx = 3500
    latent_vec = L_input_segm_mu[idx:idx+1]
    latent_vec = np.concatenate([latent_vec for i in range (35)], axis=0)
    if len(latent_vec.shape) == 1 : latent_vec = latent_vec[...,None]
    latent_vec = torch.tensor(latent_vec).to(device)
    variation_of_latent = cLDM_concat_model.generate(latent_vec, shape_data)
    variation_of_latent = format.torch_to_numpy(variation_of_latent)
    variation_of_latent = np.concatenate([variation_of_latent, L_input_mod1[idx][None,...]], axis=0)

    print("   closest images ...")
    cpt = 0
    L_closest_img = []
    nb_ = len(variation_of_latent)
    for gen_imgs in variation_of_latent :
        L_comparaison = [MSE(gen_imgs, input_img.squeeze()) for input_img in L_input_mod1]
        idx = np.argmin(L_comparaison)
        if cpt == nb_ -1 :
            L_closest_img.append(L_input_mod2[idx])
        else:
            L_closest_img.append(L_input_mod1[idx])
        cpt += 1
    L_closest_img = np.array(L_closest_img)

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Closest imgs of variations of a latent vector (train)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        if i < 35 :
            plt.imshow(L_closest_img[i,0,...], cmap="gray", vmin=0, vmax=1)
        else :
            plt.imshow(L_closest_img[i,0,...], cmap="gray", vmin=0, vmax=2)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_closest_train_imgs.png"))

    plt.figure(figsize=(20,15), constrained_layout=True)
    plt.suptitle(f"Generation of variations of a latent vector (train)")
    for i in range (36):
        plt.subplot(6,6,i+1)
        plt.imshow(variation_of_latent[i,0,...], cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        if i == 35 : plt.title("Original")
    # plt.show()
    plt.savefig(os.path.join(output_folder, f"Batch_train_imgs.png"))


    print("Rotation ...")
    name_folder = f"Rotation_segmentation_figure_paper"
    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
            os.makedirs(output_folder)


    mod_seg = AE_KL_model_seg.autoencoderKL.to(device)

    idx_ = 80
    segmentation = L_input_mod2[idx_]
    for k in range (1):
        seg_rotated = np.rot90(segmentation, k=1, axes=(1, 2))
        seg_rotated = np.copy(seg_rotated[None,...])

        seg_rotated_tmp = torch.tensor(seg_rotated).to(device)

        z_mu, z_sigma = mod_seg.encode(seg_rotated_tmp)
        z_mu = format.torch_to_numpy(z_mu)


        latent_vec = np.concatenate([z_mu for i in range (35)], axis=0)
        if len(latent_vec.shape) == 1 : latent_vec = latent_vec[...,None]
        latent_vec = torch.tensor(latent_vec).to(device)
        variation_of_latent = cLDM_concat_model.generate(latent_vec, shape_data)
        variation_of_latent = format.torch_to_numpy(variation_of_latent)
        variation_of_latent = np.concatenate([variation_of_latent, L_input_mod2[idx_][None,...]], axis=0)

        print("   closest images ...")
        cpt = 0
        L_closest_img = []
        nb_ = len(variation_of_latent)
        for gen_imgs in variation_of_latent :
            L_comparaison = [MSE(gen_imgs, input_img.squeeze()) for input_img in L_input_mod1]
            idx = np.argmin(L_comparaison)
            if cpt == nb_ -1 :
                L_closest_img.append(seg_rotated.squeeze(axis=0))
            else:
                L_closest_img.append(L_input_mod1[idx])
            cpt += 1
        L_closest_img = np.array(L_closest_img)

        plt.figure(figsize=(20,15), constrained_layout=True)
        plt.suptitle(f"Generation of variations of a latent vector (train)")
        for i in range (36):
            if i == 35: vmax=2
            else: vmax=1
            plt.subplot(6,6,i+1)
            plt.imshow(variation_of_latent[i,0,...], cmap="gray", vmin=0, vmax=vmax)
            plt.axis('off')
            if i == 35 : plt.title("Original")
        # plt.show()
        plt.savefig(os.path.join(output_folder, f"Batch_train_imgs_rot={270}°.svg"))



if __name__ == "__main__" :
    main()