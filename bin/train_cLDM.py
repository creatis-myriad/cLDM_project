import hydra
import os
import pickle
import sys
import torch

import numpy as np

sys.path.append('../') 

from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from nn_lib.common.funPath import GetPaths
from nn_lib.models.cLDM import cLDM
from nn_lib.models.AE_KL import AE_KL
from nn_lib.models.ARVAE import ARVAE
from nn_lib.models.VanillaVAE import VanillaVAE
from nn_lib.utils import format
from nn_lib.utils.pylogger import RankedLogger
from nn_lib.utils.logging_utils import log_hyperparameters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint



log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(
    version_base="1.3",
    config_path="../nn_lib/config/",
    config_name=sys.argv[1].split("=")[-1], # sys.argv[1] = +config_name=config.yaml
)
def main(cfg) :
    seed_everything(cfg.model.seed, workers=True)
    device = torch.device(cfg.model.device)
    batch_size = cfg.model.train_params.batch_size


    log.info(f"Instantiating Datamodule <{cfg.datamodule._target_}>")
    path_compress_data = GetPaths(cfg.model.path_model_autoencoder, ["output_for_ldm"])["output_for_ldm"][0]
    if cfg.model.path_model_cond :
        path_LS_cond_data  = GetPaths(cfg.model.path_model_cond, ["embedding"])["embedding"][0]
    else : path_LS_cond_data = None

    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg = cfg
    Datamodule_.path_compress_data = path_compress_data
    Datamodule_.path_cond_data = path_LS_cond_data
    Datamodule_.setup()
    shape_data = Datamodule_.train[0]["input_mu"].shape

    log.info("="*30)
    log.info("len train: "+str(len(Datamodule_.train)))
    log.info("len val  : "+str(len(Datamodule_.val)))
    log.info("len test : "+str(len(Datamodule_.test)))
    log.info("shape inp: "+str(shape_data))
    log.info("="*30)


    log.info("Instantiating callbacks")
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.train_params.callback.monitor,
        dirpath=cfg.path.output_path,
        filename='model-{epoch:02d}',
        save_last=True
    )


    log.info("Instantiating model <AutoencoderKL>")
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
            log.info("Instantiating model <VanillaVAE>")

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
            log.info("Instantiating model <ARVAE>")

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


    log.info("Instantiating model <cLDM>")
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


    log.info("Instantiating trainer <Trainer>")
    trainer = Trainer(
        default_root_dir=cfg.path.output_path,
        callbacks=[
            checkpoint_callback,
        ],
        min_epochs=cfg.model.train_params.min_epoch,
        max_epochs=cfg.model.train_params.max_epoch,
        log_every_n_steps=int(len(Datamodule_.train)/batch_size),
    )


    log.info("Logging hyperparameters!")
    object_dict = {
        "cfg": cfg,
        "datamodule": Datamodule_,
        "model": cLDM_model,
        "trainer": trainer,
    }
    log_hyperparameters(object_dict)


    log.info("Training ...")
    trainer.fit(cLDM_model, Datamodule_)


    log.info("Generating Images...")
    sum_=0
    D_img_generated = {
        "img_generated": [],
        "vec_conditioning": [],
    }
    nb_data = len(Datamodule_.test)
    for data in Datamodule_.test_dataloader():
        conditioning = torch.Tensor(data["input_cond"]).to(device)
        decoded_img = cLDM_model.generate(conditioning, shape_data)
        decoded_img = format.torch_to_numpy(decoded_img)

        for batch_decoded, batch_cond in zip(decoded_img, data["input_cond"]) :
            D_img_generated["img_generated"].append(batch_decoded)
            D_img_generated["vec_conditioning"].append(format.torch_to_numpy(batch_cond))

        sum_+=len(data["input_cond"])
        log.info(f"  nb generated: {sum_}/{nb_data} imgs -- {(sum_/nb_data)*100:.2f} %")
        if sum_ > 500: break


    log.info("Saving Generated Images...")
    with open(os.path.join(cfg.path.output_path, f"D_img_generated.pkl"), 'wb') as handle:
        pickle.dump(D_img_generated, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    log.info("done")



if __name__ == "__main__" :
    main()


