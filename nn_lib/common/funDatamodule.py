import os
import pickle

import numpy as np

from hydra.utils import instantiate
from nn_lib.common.funPath import GetPaths
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset



class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=None,
        num_workers=None,
        persistent_workers=None,
        split_test=None,
        split_val=None,
        shuffle=None,
        ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.split_test = split_test
        self.split_val  = split_val
        self.shuffle    = shuffle
        
        self.train = None
        self.val   = None
        self.test  = None

    def setup(self, stage=None): pass

    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        concat_data = ConcatDataset([self.train, self.val, self.test])
        return DataLoader(
            concat_data, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )


class LgeMyosaiq_v2_Datamodule(DataModule):
    def __init__(
            self,
            cfg=None,
            paths=None,
            name_files=None,
            **kwargs,
        ):
        
        super().__init__(**kwargs)
        self.cfg = cfg
        self.paths = paths
        self.name_files = name_files
        
    def setup(self, stage=None):
        L_pat_name = sorted(os.listdir(self.paths))
        idx_test = int(len(L_pat_name)*(1-self.split_test))
        idx_val  = int(len(L_pat_name[:idx_test])*(1-self.split_val))

        L_paths_pat_name_train = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[:idx_val]]
        L_paths_pat_name_val   = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_val:idx_test]]
        L_paths_pat_name_test  = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_test:]]

        D_path_train = self.get_D_paths(L_paths_pat_name_train)
        D_path_val   = self.get_D_paths(L_paths_pat_name_val)
        D_path_test  = self.get_D_paths(L_paths_pat_name_test)

        self.train = instantiate(
            self.cfg.dataset,
            paths=D_path_train,
        )
        self.val = instantiate(
            self.cfg.dataset,
            paths=D_path_val,
        )
        self.test = instantiate(
            self.cfg.dataset,
            paths=D_path_test,
        )
        return

    def get_D_paths(self, L_paths_pat_name):
        D_paths = {}
        for name_f in self.name_files : D_paths[name_f] = []

        for paths_pat_name in L_paths_pat_name :
            D_path_patient = GetPaths(paths_pat_name, self.name_files)
            for key, path in D_path_patient.items() :
                D_paths[key] += sorted(path)
        return D_paths


class CompressLgeSegDatamodule(DataModule):
    def __init__(
            self,
            cfg=None,
            paths=None,
            name_files=None,
            path_compress_data=None,
            nb_compress_train=None,
            nb_compress_val=None,
            **kwargs,
        ):
        
        super().__init__(**kwargs)
        self.cfg = cfg
        self.paths = paths
        self.name_files = name_files
        self.path_compress_data = path_compress_data
        self.nb_compress_train = nb_compress_train
        self.nb_compress_val = nb_compress_val
        
    def setup(self, stage=None):
        L_pat_name = sorted(os.listdir(self.paths))
        idx_test = int(len(L_pat_name)*(1-self.split_test))
        idx_val  = int(len(L_pat_name[:idx_test])*(1-self.split_val))

        L_paths_pat_name_train = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[:idx_val]]
        L_paths_pat_name_val   = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_val:idx_test]]
        L_paths_pat_name_test  = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_test:]]

        D_path_train = self.get_D_paths(L_paths_pat_name_train)
        D_path_val   = self.get_D_paths(L_paths_pat_name_val)
        D_path_test  = self.get_D_paths(L_paths_pat_name_test)


        idx_train_compress_cond = self.nb_compress_train
        idx_val_compress_cond   = self.nb_compress_val

        with open(self.path_compress_data, "rb") as f : compress_data = pickle.load(f)
        comp_data_mu = compress_data["z_mu"]
        comp_data_mu_train = comp_data_mu[:idx_train_compress_cond]
        comp_data_mu_val   = comp_data_mu[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_data_mu_test  = comp_data_mu[idx_train_compress_cond+idx_val_compress_cond:]

        comp_data_sig = compress_data["z_sigma"]
        comp_data_sig_train = comp_data_sig[:idx_train_compress_cond]
        comp_data_sig_val   = comp_data_sig[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_data_sig_test  = comp_data_sig[idx_train_compress_cond+idx_val_compress_cond:]


        self.train = instantiate(
            self.cfg.dataset,
            paths=D_path_train,
            compress_data_mu=comp_data_mu_train,
            compress_data_sig=comp_data_sig_train,
        )
        self.val = instantiate(
            self.cfg.dataset,
            paths=D_path_val,
            compress_data_mu=comp_data_mu_val,
            compress_data_sig=comp_data_sig_val,
        )
        self.test = instantiate(
            self.cfg.dataset,
            paths=D_path_test,
            compress_data_mu=comp_data_mu_test,
            compress_data_sig=comp_data_sig_test,
        )
        return

    def get_D_paths(self, L_paths_pat_name):
        D_paths = {}
        for name_f in self.name_files : D_paths[name_f] = []

        for paths_pat_name in L_paths_pat_name :
            D_path_patient = GetPaths(paths_pat_name, self.name_files)
            for key, path in D_path_patient.items() :
                D_paths[key] += sorted(path)
        return D_paths


class CompressLgeSegCondDatamodule(DataModule):
    def __init__(
            self,
            cfg=None,
            paths=None,
            name_files=None,
            path_compress_data=None,
            path_cond_data=None,
            nb_compress_train=None,
            nb_compress_val=None,
            **kwargs,
        ):
        
        super().__init__(**kwargs)
        self.cfg = cfg
        self.paths = paths
        self.name_files = name_files
        self.path_compress_data = path_compress_data
        self.path_cond_data = path_cond_data
        self.nb_compress_train = nb_compress_train
        self.nb_compress_val = nb_compress_val
        
    def setup(self, stage=None):
        L_pat_name = sorted(os.listdir(self.paths))
        idx_test = int(len(L_pat_name)*(1-self.split_test))
        idx_val  = int(len(L_pat_name[:idx_test])*(1-self.split_val))

        L_paths_pat_name_train = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[:idx_val]]
        L_paths_pat_name_val   = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_val:idx_test]]
        L_paths_pat_name_test  = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_test:]]

        D_path_train = self.get_D_paths(L_paths_pat_name_train)
        D_path_val   = self.get_D_paths(L_paths_pat_name_val)
        D_path_test  = self.get_D_paths(L_paths_pat_name_test)


        idx_train_compress_cond = self.nb_compress_train
        idx_val_compress_cond   = self.nb_compress_val

        with open(self.path_compress_data, "rb") as f : compress_data = pickle.load(f)
        comp_data_mu = compress_data["z_mu"]
        comp_data_mu_train = comp_data_mu[:idx_train_compress_cond]
        comp_data_mu_val   = comp_data_mu[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_data_mu_test  = comp_data_mu[idx_train_compress_cond+idx_val_compress_cond:]

        comp_data_sig = compress_data["z_sigma"]
        comp_data_sig_train = comp_data_sig[:idx_train_compress_cond]
        comp_data_sig_val   = comp_data_sig[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_data_sig_test  = comp_data_sig[idx_train_compress_cond+idx_val_compress_cond:]


        with open(self.path_cond_data, "rb") as f : cond_data = pickle.load(f)
        if type(cond_data) == dict : cond_data = cond_data["emb_shared"]
        else : cond_data = cond_data
        cond_data_train = cond_data[:idx_train_compress_cond]
        cond_data_val   = cond_data[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        cond_data_test  = cond_data[idx_train_compress_cond+idx_val_compress_cond:]


        self.train = instantiate(
            self.cfg.dataset,
            paths=D_path_train,
            compress_data_mu=comp_data_mu_train,
            compress_data_sig=comp_data_sig_train,
            cond_data=cond_data_train,
        )
        self.val = instantiate(
            self.cfg.dataset,
            paths=D_path_val,
            compress_data_mu=comp_data_mu_val,
            compress_data_sig=comp_data_sig_val,
            cond_data=cond_data_val,
        )
        self.test = instantiate(
            self.cfg.dataset,
            paths=D_path_test,
            compress_data_mu=comp_data_mu_test,
            compress_data_sig=comp_data_sig_test,
            cond_data=cond_data_test,
        )
        return

    def get_D_paths(self, L_paths_pat_name):
        D_paths = {}
        for name_f in self.name_files : D_paths[name_f] = []

        for paths_pat_name in L_paths_pat_name :
            D_path_patient = GetPaths(paths_pat_name, self.name_files)
            for key, path in D_path_patient.items() :
                D_paths[key] += sorted(path)
        return D_paths


class CompressLgeSegCond_Scalars_Datamodule(DataModule):
    def __init__(
            self,
            cfg=None,
            paths=None,
            name_files=None,
            path_compress_data=None,
            keys_cond_data=None,
            nb_compress_train=None,
            nb_compress_val=None,
            nb_cond_train=None,
            nb_cond_val=None,
            **kwargs,
        ):
        
        super().__init__(**kwargs)
        self.cfg = cfg
        self.paths = paths
        self.name_files = name_files
        self.path_compress_data = path_compress_data
        self.keys_cond_data = keys_cond_data
        self.nb_compress_train = nb_compress_train
        self.nb_compress_val = nb_compress_val
        self.nb_cond_train = nb_cond_train
        self.nb_cond_val = nb_cond_val
        
    def setup(self, stage=None):
        L_pat_name = sorted(os.listdir(self.paths))
        idx_test = int(len(L_pat_name)*(1-self.split_test))
        idx_val  = int(len(L_pat_name[:idx_test])*(1-self.split_val))

        L_paths_pat_name_train = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[:idx_val]]
        L_paths_pat_name_val   = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_val:idx_test]]
        L_paths_pat_name_test  = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_test:]]

        D_path_train = self.get_D_paths(L_paths_pat_name_train)
        D_path_val   = self.get_D_paths(L_paths_pat_name_val)
        D_path_test  = self.get_D_paths(L_paths_pat_name_test)


        idx_train_compress = self.nb_compress_train
        idx_val_compress   = self.nb_compress_val

        with open(self.path_compress_data, "rb") as f : compress_data = pickle.load(f)
        comp_data_mu = compress_data["z_mu"]
        comp_data_mu_train = comp_data_mu[:idx_train_compress]
        comp_data_mu_val   = comp_data_mu[idx_train_compress:idx_train_compress+idx_val_compress]
        comp_data_mu_test  = comp_data_mu[idx_train_compress+idx_val_compress:]

        comp_data_sig = compress_data["z_sigma"]
        comp_data_sig_train = comp_data_sig[:idx_train_compress]
        comp_data_sig_val   = comp_data_sig[idx_train_compress:idx_train_compress+idx_val_compress]
        comp_data_sig_test  = comp_data_sig[idx_train_compress+idx_val_compress:]


        idx_train_cond = self.nb_cond_train
        idx_val_cond   = self.nb_cond_val

        with open(self.cfg.processing.path_metrics, 'rb') as file: D_metrics = pickle.load(file)

        L_cond_data = []
        for key in self.keys_cond_data : L_cond_data.append(D_metrics[key])
        L_cond_train = {
            key: cond_data[:idx_train_cond] 
            for cond_data, key in zip(L_cond_data, self.keys_cond_data)
        }
        L_cond_val   = {
            key: cond_data[idx_train_cond:idx_train_cond+idx_val_cond] 
            for cond_data, key in zip(L_cond_data, self.keys_cond_data)
        }
        L_cond_test  = {
            key: cond_data[idx_train_cond+idx_val_cond:] 
            for cond_data, key in zip(L_cond_data, self.keys_cond_data)
        }


        self.train = instantiate(
            self.cfg.dataset,
            paths=D_path_train,
            compress_data_mu=comp_data_mu_train,
            compress_data_sig=comp_data_sig_train,
            cond_data=L_cond_train,
        )
        self.val = instantiate(
            self.cfg.dataset,
            paths=D_path_val,
            compress_data_mu=comp_data_mu_val,
            compress_data_sig=comp_data_sig_val,
            cond_data=L_cond_val,
        )
        self.test = instantiate(
            self.cfg.dataset,
            paths=D_path_test,
            compress_data_mu=comp_data_mu_test,
            compress_data_sig=comp_data_sig_test,
            cond_data=L_cond_test,
        )
        return

    def get_D_paths(self, L_paths_pat_name):
        D_paths = {}
        for name_f in self.name_files : D_paths[name_f] = []

        for paths_pat_name in L_paths_pat_name :
            D_path_patient = GetPaths(paths_pat_name, self.name_files)
            for key, path in D_path_patient.items() :
                D_paths[key] += sorted(path)
        return D_paths


class CompressLgeSeg_CNCond_Datamodule(DataModule):
    def __init__(
            self,
            cfg=None,
            paths=None,
            name_files=None,
            path_compress_data=None,
            path_compress_segm=None,
            nb_compress_train=None,
            nb_compress_val=None,
            **kwargs,
        ):
        
        super().__init__(**kwargs)
        self.cfg = cfg
        self.paths = paths
        self.name_files = name_files
        self.path_compress_data = path_compress_data
        self.path_compress_segm = path_compress_segm
        self.nb_compress_train = nb_compress_train
        self.nb_compress_val = nb_compress_val
        
    def setup(self, stage=None):
        L_pat_name = sorted(os.listdir(self.paths))
        idx_test = int(len(L_pat_name)*(1-self.split_test))
        idx_val  = int(len(L_pat_name[:idx_test])*(1-self.split_val))

        L_paths_pat_name_train = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[:idx_val]]
        L_paths_pat_name_val   = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_val:idx_test]]
        L_paths_pat_name_test  = [os.path.join(self.paths, pat_name) for pat_name in L_pat_name[idx_test:]]

        D_path_train = self.get_D_paths(L_paths_pat_name_train)
        D_path_val   = self.get_D_paths(L_paths_pat_name_val)
        D_path_test  = self.get_D_paths(L_paths_pat_name_test)


        idx_train_compress_cond = self.nb_compress_train
        idx_val_compress_cond   = self.nb_compress_val

        with open(self.path_compress_data, "rb") as f : compress_data = pickle.load(f)
        comp_data_mu = compress_data["z_mu"]
        comp_data_mu_train = comp_data_mu[:idx_train_compress_cond]
        comp_data_mu_val   = comp_data_mu[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_data_mu_test  = comp_data_mu[idx_train_compress_cond+idx_val_compress_cond:]

        comp_data_sig = compress_data["z_sigma"]
        comp_data_sig_train = comp_data_sig[:idx_train_compress_cond]
        comp_data_sig_val   = comp_data_sig[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_data_sig_test  = comp_data_sig[idx_train_compress_cond+idx_val_compress_cond:]

        with open(self.path_compress_segm, "rb") as f : compress_segm = pickle.load(f)
        comp_segm_mu = compress_segm["z_mu"]
        comp_segm_mu_train = comp_segm_mu[:idx_train_compress_cond]
        comp_segm_mu_val   = comp_segm_mu[idx_train_compress_cond:idx_train_compress_cond+idx_val_compress_cond]
        comp_segm_mu_test  = comp_segm_mu[idx_train_compress_cond+idx_val_compress_cond:]


        self.train = instantiate(
            self.cfg.dataset,
            paths=D_path_train,
            compress_data_mu=comp_data_mu_train,
            compress_data_sig=comp_data_sig_train,
            compress_segm_mu=comp_segm_mu_train,
        )
        self.val = instantiate(
            self.cfg.dataset,
            paths=D_path_val,
            compress_data_mu=comp_data_mu_val,
            compress_data_sig=comp_data_sig_val,
            compress_segm_mu=comp_segm_mu_val,
        )
        self.test = instantiate(
            self.cfg.dataset,
            paths=D_path_test,
            compress_data_mu=comp_data_mu_test,
            compress_data_sig=comp_data_sig_test,
            compress_segm_mu=comp_segm_mu_test,
        )
        return

    def get_D_paths(self, L_paths_pat_name):
        D_paths = {}
        for name_f in self.name_files : D_paths[name_f] = []

        for paths_pat_name in L_paths_pat_name :
            D_path_patient = GetPaths(paths_pat_name, self.name_files)
            for key, path in D_path_patient.items() :
                D_paths[key] += sorted(path)
        return D_paths



