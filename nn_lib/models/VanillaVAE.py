import torch

from nn_lib.models._Layers.Sampling import reparameterize
from nn_lib.models._Loss.funLoss import kl_div, MSE, CrossEntropy
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional.regression import mean_absolute_percentage_error



class VanillaVAE(LightningModule):
    def __init__(
            self, 
            config=None,
            encoder=None,
            decoder=None,
            lat_dim=None,
            alpha=None,
            beta=None,
            learning_rate=None
        ):

        super().__init__()
        self.cfg = config
        self.encoder = encoder
        self.decoder = decoder
        self.lat_dim = lat_dim
        self.alpha = alpha
        self.beta = beta
        self.lr = learning_rate

        self.fc_mu = nn.Linear(lat_dim, lat_dim)
        self.fc_var = nn.Linear(lat_dim, lat_dim)

        self.save_hyperparameters(
            ignore=["encoder","decoder",],
        )

    def forward(self, x):
        z = self.encoder(x)
        mu  = self.fc_mu(z)
        var = self.fc_var(z)
        z = reparameterize(mu, var)
        
        D_outputs = {
            "mu": mu, 
            "var": var, 
            "z": z,
        }

        x_hat = self.decoder(z)
        if "segmentation" in self.cfg.architecture.decoders.keys():
            D_outputs["x_hat_classes"] = x_hat
            x_hat = torch.argmax(x_hat, dim=1, keepdim=True).to(mu.device)
        D_outputs["x_hat"] = x_hat
        return D_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def compute_loss(self, x, D_outputs):
        if "segmentation" in self.cfg.architecture.decoders.keys():
            elbo = CrossEntropy(
                x, 
                D_outputs["x_hat_classes"], 
                weight_classes=self.cfg.model.net.weight_classes,
            )*self.alpha
        else:
            elbo = MSE(x, D_outputs["x_hat"])*self.alpha
        KL = kl_div(D_outputs["mu"], D_outputs["var"])*self.beta
        loss = elbo + KL
        return {"loss":loss, "elbo":elbo, "KL":KL}

    def training_step(self, train_batch, batch_idx):
        x = train_batch["input"]
        D_outputs = self.forward(x)
        D_loss = self.compute_loss(x, D_outputs)

        self.log(
            'elbo_loss', 
            D_loss["elbo"],
            on_step=False, 
            on_epoch=True,
        )
        self.log(
            'KL_loss', 
            D_loss["KL"],
            on_step=False, 
            on_epoch=True,
        )
        self.log(
            'train_loss', 
            D_loss["loss"],  
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )
        return D_loss["loss"]

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["input"]
        D_outputs = self.forward(x)
        D_loss = self.compute_loss(x, D_outputs)
        self.log(
            'val_loss',
            D_loss["loss"],  
            on_step=False, 
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx): 
        x = batch["input"]
        D_outputs = self.forward(x)
        D_loss = self.compute_loss(x, D_outputs)
        self.log(
            'test_loss', 
            D_loss["elbo"],
        )
        self.log(
            'MAPE (%)', 
            mean_absolute_percentage_error(x, D_outputs["x_hat"])*100
        )

    def predict_step(self, batch, batch_idx): 
        return self(batch["input"])














