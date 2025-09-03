import os
import pickle
import sys
import torch

import numpy as np

sys.path.append('../') 

from math import log10, sqrt 
from hydra.utils import instantiate
from nn_lib.common.funPath import GetPaths
from nn_lib.models.AE_KL import AE_KL
from nn_lib.utils import format



def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 



def main() :
    print("Loading parameters ...")
    path_ = sys.argv[-1] # Dir of the model
    path_model = GetPaths(path_, [".ckpt"])[".ckpt"][-1]
    D_model = torch.load(path_model)
    cfg = D_model["hyper_parameters"]["config"]
    device = torch.device(cfg.model.device)


    print(f"Instantiating model <{cfg.datamodule._target_}>")
    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg=cfg
    Datamodule_.setup()
    full_data = Datamodule_.predict_dataloader()


    L_input_norm = []
    L_input_data  = []
    L_pat_name = []
    L_seq_name = []
    for sample in full_data :
        L_input_norm.append(sample["input_norm"])
        L_input_data.append(sample["input"])
        L_pat_name.append(sample["pat_name"])
        L_seq_name.append(sample["seq_name"])
    L_input_norm = np.concatenate(L_input_norm, axis=0)
    L_input_data  = np.concatenate(L_input_data, axis=0)
    L_pat_name = np.concatenate(L_pat_name, axis=0)
    L_seq_name = np.concatenate(L_seq_name, axis=0)



    print(f"Instantiating model <AE_KL>")
    autoencoderKL = instantiate(cfg.architecture.autoencoders.autoencoderKL)
    patchDiscriminator = instantiate(cfg.architecture.autoencoders.patchDiscriminator)

    AE_KL_model = AE_KL(
        config=cfg,
        autoencoderKL=autoencoderKL,
        patchDiscriminator=patchDiscriminator,
        learning_rate_generator=cfg.model.net.lr_g,
        learning_rate_discriminator=cfg.model.net.lr_d,
    ).to(device)

    AE_KL_model.load_state_dict(
        D_model["state_dict"], 
    )



    import matplotlib.pyplot as plt

    for i in range (2000,2200,1) :
        img_tensor = torch.tensor(L_input_norm[i], dtype=torch.float32).to(device)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.unsqueeze(0)

        D_out = AE_KL_model(img_tensor)
        recons_img = format.torch_to_numpy(D_out["reconstruction"]).squeeze()
        img = L_input_norm[i]


        plt.figure(figsize=(20, 15), constrained_layout=True)
        
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title("Original Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(recons_img, cmap="gray", vmin=0, vmax=1)
        plt.title("Reconstructed Image")
        

        diff = np.nansum(abs(img - recons_img))/recons_img.size
        diff = (diff*100)
        plt.subplot(1, 3, 3)
        plt.imshow(abs(img - recons_img), vmin=0, vmax=1)
        plt.title(f"Difference: {diff:.3f}% /// PSNR: {PSNR(img, recons_img):.3f}dB")
        
        plt.show()



if __name__ == "__main__" :
    main()
