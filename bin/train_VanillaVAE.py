import hydra
import os
import pickle
import sys
import torch

import numpy as np

sys.path.append('../') 

from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from nn_lib.models.VanillaVAE import VanillaVAE
from nn_lib.utils import format
from nn_lib.utils.pylogger import RankedLogger
from nn_lib.utils.logging_utils import log_hyperparameters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(
    version_base="1.3",
    config_path="../nn_lib/config/",
    config_name=sys.argv[1].split("=")[-1], # sys.argv[1] = +config_name=config.yaml
)
def main(cfg) :
    seed_everything(cfg.model.seed, workers=True)
    device = torch.device(cfg.model.device)


    log.info(f"Instantiating Datamodule <{cfg.datamodule._target_}>")
    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg = cfg
    Datamodule_.setup()

    log.info("="*30)
    log.info("len train: "+str(len(Datamodule_.train)))
    log.info("len val  : "+str(len(Datamodule_.val)))
    log.info("len test : "+str(len(Datamodule_.test)))
    log.info("="*30)


    log.info("Instantiating callbacks")
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.train_params.callback.monitor,
        dirpath=cfg.path.output_path,
        filename='model-{epoch:02d}',
        save_top_k=1,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.model.train_params.callback.monitor, 
        min_delta=cfg.model.train_params.callback.min_delta, 
        patience=cfg.model.train_params.callback.patience, 
        check_on_train_epoch_end=True,
        mode='min', 
    )


    log.info("Instantiating model <VanillaVAE>")
    lat_dims = cfg.model.net.lat_dims

    enc = instantiate(cfg.architecture.encoders).to(device)
    dec = instantiate(cfg.architecture.decoders).to(device)

    VAE_model = VanillaVAE(
        config=cfg,
        encoder=enc,
        decoder=dec,
        lat_dim=lat_dims,
        learning_rate=cfg.model.net.lr,
        alpha=cfg.model.net.alpha,
        beta=cfg.model.net.beta,
    ).to(device)


    log.info("Instantiating trainer <Trainer>")
    trainer = Trainer(
        default_root_dir=cfg.path.output_path,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        min_epochs=cfg.model.train_params.min_epoch,
        max_epochs=cfg.model.train_params.max_epoch,
        log_every_n_steps=int(len(Datamodule_.train)/cfg.model.train_params.batch_size),
    )


    log.info("Logging hyperparameters!")
    object_dict = {
        "cfg": cfg,
        "datamodule": Datamodule_,
        "model": VAE_model,
        "trainer": trainer,
    }
    log_hyperparameters(object_dict)


    log.info("Training ...")
    trainer.fit(VAE_model, Datamodule_)


    log.info("Testing ...")
    result = trainer.test(
        model=VAE_model,
        dataloaders=Datamodule_,
        ckpt_path="best",
    )
    log.info("test_loss = "+str(result[0]["test_loss"]))
    log.info("MAPE (%)  = "+str(result[0]["MAPE (%)"]))


    log.info("Saving embedding ...")
    emb = []
    prediction = trainer.predict(VAE_model, Datamodule_)
    for data_emb in prediction :
        D_outputs = data_emb
        for mu in D_outputs["mu"] : 
            emb.append(format.torch_to_numpy(mu))
    emb = np.array(emb, dtype=np.float32)

    with open(os.path.join(cfg.path.output_path, "embedding.pkl"), 'wb') as handle:
        pickle.dump(emb, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    log.info("done")



if __name__ == "__main__" :
    main()


