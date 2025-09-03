import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from nn_lib.models._Loss.funLoss import MSE
from pytorch_lightning import LightningModule



class ControlNet(LightningModule):
    def __init__(
            self, 
            config=None,
            model=None,
            autoencoder_model=None, 
            LDM_model=None,
            conditional_model=None,
            learning_rate=None,
        ):
        super().__init__()
        self.cfg = config
        self.model = model
        self.autoencoder_model = autoencoder_model
        self.LDM_model = LDM_model
        self.conditional_model = conditional_model
        self.lr = learning_rate

        self.scale_factor = LDM_model.scale_factor
        self.num_timesteps = LDM_model.num_timesteps
        self.scheduler = LDM_model.scheduler

        self.save_hyperparameters(
            ignore=["model", "autoencoder_model", "LDM_model", "conditional_model"],
        )

    def forward(self, cn_cond, xt, t):
        down_block_res_samples, mid_block_res_sample = self.model(
            x=xt, timesteps=t, controlnet_cond=cn_cond
        )

        pred = self.LDM_model.model(
            x=xt,
            timesteps=t,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def compute_loss(self, noise, noise_pred):
        loss = MSE(noise, noise_pred)
        return {"loss":loss}

    def training_step(self, train_batch, batch_idx):
        z_mu, z_sig = train_batch["input_mu"], train_batch["input_sig"]
        # z_mu_seg = train_batch["segm_mu"]
        z_mu_seg = train_batch["input_norm_mod2"].unsqueeze(1)

        z_0 = z_mu + z_sig * torch.randn_like(z_sig).to(self.cfg.model.device)
        z_0 = z_0 * self.scale_factor

        noise = torch.randn_like(z_0).to(self.cfg.model.device)
        t = torch.randint(0, self.num_timesteps, (z_0.shape[0],), device=self.cfg.model.device).long()
        noisy_z_0 = self.scheduler.add_noise(original_samples=z_0, noise=noise, timesteps=t)
        noise_pred = self(z_mu_seg, noisy_z_0, t)

        D_loss = self.compute_loss(noise, noise_pred)
        self.log(
            'train_loss', 
            D_loss["loss"],  
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )
        return D_loss["loss"]

    def validation_step(self, batch, batch_idx):
        z_mu, z_sig = batch["input_mu"], batch["input_sig"]
        # z_mu_seg = train_batch["segm_mu"]
        z_mu_seg = batch["input_norm_mod2"].unsqueeze(1)


        z_0 = z_mu + z_sig * torch.randn_like(z_sig).to(self.cfg.model.device)
        z_0 = z_0 * self.scale_factor

        noise = torch.randn_like(z_0).to(self.cfg.model.device)
        t = torch.randint(0, self.num_timesteps, (z_0.shape[0],), device=self.cfg.model.device).long()
        noisy_z_0 = self.scheduler.add_noise(original_samples=z_0, noise=noise, timesteps=t)
        noise_pred = self(z_mu_seg, noisy_z_0, t)
        
        # Save images on validation step
        path_output = os.path.join(self.cfg.path.output_path, "Validation_Images")
        if not os.path.exists(path_output): os.mkdir(path_output) 
        if self.cfg.model.train_params.max_epoch <=100: modulo=5
        else: modulo = self.cfg.model.train_params.max_epoch//100
        if self.current_epoch%modulo==0 and batch_idx == 0:
            seg_cond = batch["input_norm_mod2"][0].cpu().numpy()
            L_imgs = self.generate_steps(z_mu_seg[0:1], noise_pred.shape[1:])
            chain = torch.cat(L_imgs, dim=-1)
            plt.style.use("default")
            plt.subplot(211)
            plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
            plt.subplot(212)
            plt.imshow(seg_cond, vmin=0, vmax=2, cmap="gray")
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(os.path.join(path_output, f"img_val_{self.current_epoch}.svg"))

        D_loss = self.compute_loss(noise, noise_pred)
        self.log(
            'val_loss', 
            D_loss["loss"],  
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )
        return D_loss["loss"]
        
    def generate(self, cn_cond, shape_data):
        xt = torch.randn(cn_cond.shape[0], *shape_data).to(self.cfg.model.device)
        mod_autoenc = self.autoencoder_model.autoencoderKL.to(self.cfg.model.device)
        self.model = self.model.to(self.cfg.model.device)
        self.LDM_model.model = self.LDM_model.model.to(self.cfg.model.device)

        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                noise_pred = self(
                    cn_cond=cn_cond,
                    xt=xt,
                    t=torch.Tensor((t,)).to(self.cfg.model.device).long(),
                )
                xt, _ = self.scheduler.step(noise_pred, t, xt)
            decoded_img = mod_autoenc.decode(xt / self.scale_factor)
        return decoded_img

    def generate_steps(self, cn_cond, shape_data):
        L_step_gen_img = []
        xt = torch.randn(1, *shape_data).to(self.cfg.model.device)
        mod_autoenc = self.autoencoder_model.autoencoderKL.to(self.cfg.model.device)
        self.model = self.model.to(self.cfg.model.device)
        self.LDM_model.model = self.LDM_model.model.to(self.cfg.model.device)

        step_save = np.arange(0,self.num_timesteps, 100)
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                noise_pred = self(
                    cn_cond=cn_cond,
                    xt=xt,
                    t=torch.Tensor((t,)).to(self.cfg.model.device).long(),
                )
                xt, _ = self.scheduler.step(noise_pred, t, xt)
                if t == 999 or t in step_save: 
                    decoded_img = mod_autoenc.decode(xt / self.scale_factor)
                    L_step_gen_img.append(decoded_img)
        return L_step_gen_img
    
    
