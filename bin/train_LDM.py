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
from nn_lib.models.LDM import LDM
from nn_lib.models.AE_KL import AE_KL
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

    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg = cfg
    Datamodule_.path_compress_data = path_compress_data
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

    for first_batch in Datamodule_.predict_dataloader():
        z = AE_KL_model.autoencoderKL.encode_stage_2_inputs(first_batch["input_mod1"].to(device))
        log.info(f"Scaling factor set to {1/torch.std(z)}")
        break
    scale_factor = 1 / format.torch_to_numpy(torch.std(z))


    log.info("Instantiating model <LDM>")
    LDM_model = LDM(
        config=cfg,
        model=instantiate(cfg.architecture.unets), 
        autoencoder_model=AE_KL_model,
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
        "model": LDM_model,
        "trainer": trainer,
    }
    log_hyperparameters(object_dict)


    log.info("Training ...")
    trainer.fit(LDM_model, Datamodule_)


    log.info("Generating Images...")
    sum_=0
    L_generated_imgs = []
    nb_data = len(Datamodule_.train)+len(Datamodule_.val)+len(Datamodule_.test)
    if nb_data > 500: nb_data = 500
    L_nb_batch = [batch_size for _ in range(nb_data//batch_size)]+[nb_data-(nb_data//batch_size)*batch_size]
    for nb_batch in L_nb_batch:
        xt = LDM_model.generate(batch=nb_batch, shape_data=shape_data)
        xt = format.torch_to_numpy(xt)
        L_generated_imgs.append(xt)
        sum_ += nb_batch
        print(f"nb generated: {sum_}/{nb_data} imgs -- {(sum_/nb_data)*100:.2f} %")


    log.info("Saving Generated Images...")
    cpt = 0
    os.mkdir(os.path.join(cfg.path.output_path, "Generated_Images"))
    for batch_gen_img in L_generated_imgs :
        for gen_img in batch_gen_img :
            with open(os.path.join(cfg.path.output_path, f"Generated_Images/img_{cpt}.pkl"), 'wb') as handle:
                pickle.dump(gen_img.squeeze(), handle, protocol=pickle.HIGHEST_PROTOCOL)  
            cpt+=1


    log.info("done")



if __name__ == "__main__" :
    main()


