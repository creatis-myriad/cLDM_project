import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from nn_lib.models._Loss.funLoss import MSE
from nn_lib.models._Scheduler.ddpm import DDPMScheduler
from pytorch_lightning import LightningModule



class cLDM_concat(LightningModule):
    def __init__(
            self, 
            config=None,
            model=None,
            autoencoder_model=None, 
            conditional_model=None,
            learning_rate=None,
            num_timesteps=None, 
            scheduler_type=None,
            scale_factor=None,
        ):
        super().__init__()
        self.cfg = config
        self.model = model
        self.autoencoder_model = autoencoder_model
        self.conditional_model = conditional_model
        self.lr = learning_rate
        self.num_timesteps = num_timesteps
        self.scheduler_type = scheduler_type
        self.scale_factor = scale_factor

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            schedule=scheduler_type,
        )


        self.save_hyperparameters(
            ignore=["model", "autoencoder_model", "conditional_model"],
        )

    def forward(self, xt, t):
        return self.model(x=xt, timesteps=t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def compute_loss(self, noise, noise_pred):
        loss = MSE(noise, noise_pred)
        return {"loss":loss}

    def training_step(self, train_batch, batch_idx):
        conditioning = train_batch["segm_mu"]
        z_mu, z_sig = train_batch["input_mu"], train_batch["input_sig"]
        z_0 = z_mu + z_sig * torch.randn_like(z_sig).to(self.cfg.model.device)
        z_0 = z_0 * self.scale_factor

        noise = torch.randn_like(z_0).to(self.cfg.model.device)
        t = torch.randint(0, self.num_timesteps, (z_0.shape[0],), device=self.cfg.model.device).long()
        noisy_z_0 = self.scheduler.add_noise(original_samples=z_0, noise=noise, timesteps=t)
        cond_noisy_z_0 = torch.cat((conditioning, noisy_z_0), dim=1)
        noise_pred = self(cond_noisy_z_0, t)

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
        conditioning = batch["segm_mu"]
        z_mu, z_sig = batch["input_mu"], batch["input_sig"]
        z_0 = z_mu + z_sig * torch.randn_like(z_sig).to(self.cfg.model.device)
        z_0 = z_0 * self.scale_factor
        
        noise = torch.randn_like(z_0).to(self.cfg.model.device)
        t = torch.randint(0, self.num_timesteps, (z_0.shape[0],), device=self.cfg.model.device).long()
        noisy_z_0 = self.scheduler.add_noise(original_samples=z_0, noise=noise, timesteps=t)
        cond_noisy_z_0 = torch.cat((conditioning, noisy_z_0), dim=1)
        noise_pred = self(cond_noisy_z_0, t)
        
        # Save images on validation step
        path_output = os.path.join(self.cfg.path.output_path, "Validation_Images")
        if not os.path.exists(path_output): os.mkdir(path_output) 
        if self.cfg.model.train_params.max_epoch <=100: modulo=5
        else: modulo = self.cfg.model.train_params.max_epoch//100
        if self.current_epoch%modulo==0 and batch_idx == 0:
            seg_cond = batch["input_norm_mod2"][0].cpu().numpy()
            L_imgs = self.generate_steps(conditioning[0:1], noise_pred.shape[1:])
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
        
    def generate(self, latent, shape_data):
        xt = torch.randn(latent.shape[0], *shape_data).to(self.cfg.model.device)
        mod = self.model.to(self.cfg.model.device)
        mod_autoenc = self.autoencoder_model.autoencoderKL.to(self.cfg.model.device)
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                xt_concat = torch.cat((latent, xt), dim=1)
                noise_pred = mod(xt_concat, torch.as_tensor(t).unsqueeze(0).to(self.cfg.model.device))
                xt, _ = self.scheduler.step(noise_pred, t, xt)
            decoded_img = mod_autoenc.decode(xt / self.scale_factor)
        return decoded_img
    

    def generate_steps(self, latent, shape_data):
        L_step_gen_img = []
        xt = torch.randn(1, *shape_data).to(self.cfg.model.device)
        mod = self.model.to(self.cfg.model.device)
        mod_autoenc = self.autoencoder_model.autoencoderKL.to(self.cfg.model.device)
        step_save = np.arange(0,self.num_timesteps, 100)
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                xt_concat = torch.cat((latent, xt), dim=1)
                noise_pred = mod(xt_concat, torch.as_tensor(t).unsqueeze(0).to(self.cfg.model.device))
                xt, _ = self.scheduler.step(noise_pred, t, xt)
                if t == 999 or t in step_save: 
                    decoded_img = mod_autoenc.decode(xt / self.scale_factor)
                    L_step_gen_img.append(decoded_img)
        return L_step_gen_img
    
    
