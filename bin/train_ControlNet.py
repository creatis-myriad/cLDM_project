import copy
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
from nn_lib.models.ControlNet import ControlNet
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
    path_compress_segm = GetPaths(cfg.model.path_model_cond, ["output_for_ldm"])["output_for_ldm"][0]

    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg = cfg
    Datamodule_.path_compress_data = path_compress_data
    Datamodule_.path_compress_segm = path_compress_segm
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


    # log.info("Instantiating model <AutoencoderKL_seg>")
    # path_AEKL_seg = cfg.model.path_model_cond
    # path_model_AEKL_seg = GetPaths(path_AEKL_seg, [".ckpt"])[".ckpt"][-1]
    # D_model_AEKL_seg = torch.load(path_model_AEKL_seg)
    # cfg_AEKL_seg = D_model_AEKL_seg["hyper_parameters"]["config"]

    # autoencoderKL = instantiate(cfg_AEKL_seg.architecture.autoencoders.autoencoderKL)
    # patchDiscriminator = instantiate(cfg_AEKL_seg.architecture.autoencoders.patchDiscriminator)

    # AE_KL_model_seg = AE_KL(
    #     config=cfg_AEKL_seg,
    #     autoencoderKL=autoencoderKL,
    #     patchDiscriminator=patchDiscriminator,
    #     learning_rate_generator=cfg_AEKL_seg.model.net.lr_g,
    #     learning_rate_discriminator=cfg_AEKL_seg.model.net.lr_d,
    # ).to(device)

    # AE_KL_model_seg.load_state_dict(
    #     D_model_AEKL_seg["state_dict"], 
    # )
    # for param in AE_KL_model_seg.parameters(): param.requires_grad = False


    log.info("Instantiating model <LDM>")
    path_LDM = cfg.model.path_model_LDM
    path_model_LDM = GetPaths(path_LDM, [".ckpt"])[".ckpt"][-1]
    D_model_LDM = torch.load(path_model_LDM)
    cfg_LDM = D_model_LDM["hyper_parameters"]["config"]

    LDM_model = LDM(
        config=cfg_LDM,
        model=instantiate(cfg_LDM.architecture.unets), 
        autoencoder_model=AE_KL_model,
        learning_rate=cfg_LDM.model.net.lr,
        num_timesteps=cfg_LDM.model.net.num_timesteps, 
        scheduler_type=cfg_LDM.model.net.scheduler,
        scale_factor=cfg_LDM.model.net.scale_factor,
    ).to(device)

    LDM_model_copy = copy.deepcopy(LDM_model)
    LDM_model.load_state_dict(
        D_model_LDM["state_dict"], 
    )
    for param in LDM_model.parameters(): param.requires_grad = False


    ControlNet_model = ControlNet(
        config=cfg,
        model=instantiate(cfg.architecture.unets),
        LDM_model=LDM_model,
        autoencoder_model=AE_KL_model,
        # conditional_model=AE_KL_model_seg,
        learning_rate=cfg.model.net.lr,
    )
    ControlNet_model.load_state_dict(LDM_model_copy.state_dict(), strict=False)
    ControlNet_model = ControlNet_model.to(device)


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
        "model": ControlNet_model,
        "trainer": trainer,
    }
    log_hyperparameters(object_dict)


    log.info("Training ...")
    trainer.fit(ControlNet_model, Datamodule_)


    log.info("Generating Images...")
    sum_=0
    D_img_generated = {
        "img_generated": [],
        "vec_conditioning": [],
        "img_conditioning": [],
        "scalars_cond": [],
    }
    nb_data = len(Datamodule_.test)
    keys_metrics =  [
        "z_vals",
        "transmurality",
        "endo_surface_length",
        "infarct_size_2D"
    ]
    # mod_seg = AE_KL_model_seg.autoencoderKL.to(device)
    for data in Datamodule_.test_dataloader():
        # conditioning = torch.Tensor(data["segm_mu"]).to(device)
        conditioning = torch.Tensor(data["input_norm_mod2"]).unsqueeze(1).to(device)
        decoded_img = ControlNet_model.generate(conditioning, shape_data)
        decoded_img = format.torch_to_numpy(decoded_img)

        # decoded_cond = mod_seg.decode(conditioning / cfg_LDM.model.net.scale_factor)
        # decoded_cond = format.torch_to_numpy(decoded_cond)
        decoded_cond = data["input_norm_mod2"]

        L_metrics  = []
        for key in keys_metrics:
            L_metrics.append(data["metrics"][key])
        L_metrics = np.array(L_metrics).T

        for batch_decoded, batch_cond, batch_cond_seg, scalar_cond in zip(decoded_img, data["segm_mu"], decoded_cond, L_metrics) :
            D_img_generated["img_generated"].append(batch_decoded)
            D_img_generated["vec_conditioning"].append(format.torch_to_numpy(batch_cond))
            D_img_generated["img_conditioning"].append(format.torch_to_numpy(batch_cond_seg))
            D_img_generated["scalars_cond"].append(scalar_cond)

        sum_+=len(data["segm_mu"])
        log.info(f"  nb generated: {sum_}/{nb_data} imgs -- {(sum_/nb_data)*100:.2f} %")
        if sum_ > 500: break


    log.info("Saving Generated Images...")
    with open(os.path.join(cfg.path.output_path, f"D_img_generated.pkl"), 'wb') as handle:
        pickle.dump(D_img_generated, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    log.info("done")



if __name__ == "__main__" :
    main()


