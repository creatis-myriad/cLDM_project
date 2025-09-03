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
from nn_lib.models.LDM import LDM
from nn_lib.models.AE_KL import AE_KL
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
    # gen_images_folder = os.path.join(path_, "Generated_Images")
    device = torch.device(cfg.model.device)


    print(f"Instantiating Datamodule <{cfg.datamodule._target_}>")
    path_compress_data = GetPaths(cfg.model.path_model_autoencoder, ["output_for_ldm"])["output_for_ldm"][0]

    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg = cfg
    Datamodule_.path_compress_data = path_compress_data
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


    print("Instantiating model <LDM>")
    LDM_model = LDM(
        config=cfg,
        model=instantiate(cfg.architecture.unets), 
        autoencoder_model=AE_KL_model,
        learning_rate=cfg.model.net.lr,
        num_timesteps=cfg.model.net.num_timesteps, 
        scheduler_type=cfg.model.net.scheduler,
        scale_factor=cfg.model.net.scale_factor,
    ).to(device)
    
    LDM_model.load_state_dict(
        D_model_DDPM["state_dict"], 
    )


    print()
    print()
    print("Figures for validating Unconditional Latent Diffusion Model\n")
    name_nn = sys.argv[-1].split("/")[-1]
    output_path = "/home/deleat/Documents/RomainD/Working_space/Figures_Resultats/Results_LLDM_project/validation_inconditionnel_LDM/"
    output_path = os.path.join(output_path, name_nn)


    nb_loop = 5

    print("Generated images ...")
    name_folder = f"batch of generated imgs"
    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
    for nb_batch in range(nb_loop) :
        print(f"Batch {nb_batch}")
        batch = 36
        L_gen_imgs = LDM_model.generate(batch, shape_data)
        L_gen_imgs = format.torch_to_numpy(L_gen_imgs)
        L_gen_imgs = L_gen_imgs.squeeze()

        plt.figure(figsize=(20,15), constrained_layout=True)
        for i in range(batch) :
            plt.subplot(6,6,i+1)
            plt.imshow(L_gen_imgs[i], cmap="gray")
            plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(output_folder, f"batch {nb_batch}.png"))


    print("Comparaison distribution ...")
    name_folder = f"Test comparaison distribution"
    output_folder = os.path.join(output_path, name_folder)
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
    for nb_batch in range(nb_loop) :
        print(f"Batch {nb_batch}")
        batch = 18
        L_gen_imgs = LDM_model.generate(batch, shape_data)
        L_gen_imgs = format.torch_to_numpy(L_gen_imgs)
        L_gen_imgs = L_gen_imgs.squeeze()

        print("   closest images ...")
        L_closest_img = []
        for gen_imgs in L_gen_imgs :
            L_comparaison = [MSE(gen_imgs, input_img.squeeze()) for input_img in L_input_mod1]
            idx = np.argmin(L_comparaison)
            L_closest_img.append(L_input_mod1[idx])
        L_closest_img = np.array(L_closest_img)

        plt.figure(figsize=(20, 15), constrained_layout=True)
        for i in range(2*batch):
            plt.subplot(6, 6, i + 1)
            if i % 2 == 0:
                plt.imshow(L_gen_imgs[i//2], cmap="gray", vmin=0, vmax=1)
                plt.title("Generated")
            else:
                plt.imshow(L_closest_img[i//2,0], cmap="gray", vmin=0, vmax=1)
                plt.title("Closest")
            plt.axis("off")
        plt.savefig(os.path.join(output_folder, f"batch {nb_batch}.png"))
    
        


if __name__ == "__main__" :
    main()    










