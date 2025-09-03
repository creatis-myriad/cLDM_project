import torch

from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional.regression import mean_absolute_percentage_error



class AE_KL(LightningModule):
    def __init__(
            self, 
            config=None,
            autoencoderKL=None,
            patchDiscriminator=None,
            learning_rate_generator=None,
            learning_rate_discriminator=None,
        ):

        super().__init__()
        self.cfg = config
        self.autoencoderKL = autoencoderKL
        self.patchDiscriminator = patchDiscriminator
        self.lr_g = learning_rate_generator
        self.lr_d = learning_rate_discriminator

        self.automatic_optimization = False
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
        self.l1_loss = nn.L1Loss()

        self.save_hyperparameters(
            ignore=["autoencoderKL","patchDiscriminator",],
        )

    def forward(self, x):
        # Generator part
        reconstruction, z_mu, z_sigma = self.autoencoderKL(x)
        logits_fake_gen = self.patchDiscriminator(reconstruction.contiguous().float())[-1]

        # Discriminator part
        logits_fake_dis = self.patchDiscriminator(reconstruction.contiguous().detach())[-1]
        logits_real = self.patchDiscriminator(x.contiguous().detach())[-1]

        D_outputs = {
            "reconstruction": reconstruction,
            "z_mu": z_mu,
            "z_sigma": z_sigma,
            "logits_fake_for_gen": logits_fake_gen,
            "logits_fake_for_dis": logits_fake_dis,
            "logits_real": logits_real,
        }
        return D_outputs

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.autoencoderKL.parameters(), lr=self.lr_g)
        optimizer_d = torch.optim.Adam(self.patchDiscriminator.parameters(), lr=self.lr_d)
        return optimizer_g, optimizer_d
    
    def compute_loss(self, images, D_outputs):
        # Generator part
        recons_loss = self.l1_loss(D_outputs["reconstruction"].float(), images.float())
        kl_loss = 0.5 * torch.sum(D_outputs["z_mu"].pow(2) + D_outputs["z_sigma"].pow(2) - torch.log(D_outputs["z_sigma"].pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        p_loss = self.perceptual_loss(D_outputs["reconstruction"].float(), images.float())
        generator_loss = self.adv_loss(D_outputs["logits_fake_for_gen"], target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + self.cfg.model.net.beta * kl_loss + \
                self.cfg.model.net.perceptual_weight * p_loss + \
                self.cfg.model.net.adv_weight * generator_loss
        
        # Discriminator part
        loss_d_fake = self.adv_loss(D_outputs["logits_fake_for_dis"], target_is_real=False, for_discriminator=True)
        loss_d_real = self.adv_loss(D_outputs["logits_real"], target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.cfg.model.net.adv_weight * discriminator_loss
        return {"loss":recons_loss, "gen_loss":generator_loss, "disc_loss":discriminator_loss}, loss_g, loss_d

    def training_step(self, train_batch, batch_idx):
        x = train_batch["input"]
        D_outputs = self.forward(x)
        D_loss, loss_g, loss_d = self.compute_loss(x, D_outputs)

        g_opt, d_opt = self.optimizers()
        g_opt.zero_grad(set_to_none=True)
        loss_g.backward()
        g_opt.step()
        d_opt.zero_grad(set_to_none=True)
        loss_d.backward()
        d_opt.step()


        self.log(
            'gen_', 
            D_loss["gen_loss"],
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            'disc_', 
            D_loss["disc_loss"],
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            'train_', 
            D_loss["loss"],  
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )
        return D_loss["loss"]

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["input"]
        D_outputs = self.forward(x)
        D_loss, _, _ = self.compute_loss(x, D_outputs)

        self.log(
            'val_',
            D_loss["loss"],  
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx): 
        x = batch["input"]
        D_outputs = self.forward(x)
        D_loss, _, _ = self.compute_loss(x, D_outputs)
        self.log(
            'test_loss', 
            D_loss["loss"],
        )
        self.log(
            'MAPE (%)', 
            mean_absolute_percentage_error(x, D_outputs["reconstruction"])*100
        )

    def predict_step(self, batch, batch_idx): 
        return self(batch["input"])














