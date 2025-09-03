import cv2
import imageio
import pickle
import struct
import torch

import numpy as np

from array import array
from bisect import bisect_right



class LgeMyosaiq_v2_Dataset(torch.utils.data.Dataset) :
    def __init__(
        self,
        paths=None,
        work_on=None,
        processing=None,
        ):

        self.paths = paths
        self.work_on = work_on
        self.processing = processing

        with open(self.processing.path_metrics, 'rb') as file:
            self.D_metrics = pickle.load(file)

        self.slice_counts = []
        self.slice_indices = []
        for path_dcm in self.paths[self.work_on]:
            pat_name = path_dcm.split("/")[-2]

            idx_metrics_patient = self.D_metrics["pat_name"].index(pat_name)
            slices = self.D_metrics["index_slices"][idx_metrics_patient]
            self.slice_counts.append(len(slices))
            self.slice_indices.append(slices)
        self.cumulative_slices = np.cumsum([0] + self.slice_counts)  # Include 0 for easier indexing


    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        sample = {}

        # Find volume_idx using bisect for efficiency
        volume_idx = bisect_right(self.cumulative_slices, idx) - 1
        slice_idx  = idx - self.cumulative_slices[volume_idx]

        path_dcm = self.paths["dcm.pkl"][volume_idx]
        path_roi = self.paths["roi.pkl"][volume_idx]

        with open(path_dcm, 'rb') as file:
            dcm = pickle.load(file)
        with open(path_roi, 'rb') as file:
            roi = pickle.load(file)

        sample["pat_name"] = path_dcm.split("/")[-2]
        sample["seq_name"] = dcm.type_seq

        D_metric_patient = {}
        idx_metrics_patient = self.D_metrics["pat_name"].index(sample["pat_name"])
        slice_idx_in_list   = self.D_metrics["index_slices"][idx_metrics_patient][slice_idx]

        for key, val in self.D_metrics.items() :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                D_metric_patient[key] = val[idx_metrics_patient][slice_idx]
            else:
                D_metric_patient[key] = val[idx_metrics_patient]
        sample["metrics"] = D_metric_patient

        # # dict_keys(['pat_name', 'index_slices', 'z_vals', 'transmurality', 
        # # 'endo_surface_length', 'infarct_size_2D', 'angle_junction', 
        # # 'infarct_location', 'inversion_time', 'month', 'sequence_type', 
        # # 'age', 'sex', 'lv_mass', 'lv_edv', 'ejection_fraction', 'thrombus'])

        if self.work_on == "dcm.pkl" : 
            input = dcm.dataResampled[:,:,slice_idx_in_list]
            min_max = [np.min(input), np.max(input)]
            input_norm = (input - min_max[0])/(min_max[1] - min_max[0])

        if self.work_on == "roi.pkl" : 
            MI_and_nonMI = roi.segmentsResampled[slice_idx_in_list]['non-MI'] + roi.segmentsResampled[slice_idx_in_list]['MI']
            input = np.array(MI_and_nonMI, dtype=np.float32)
            input_norm = np.copy(input)
        
        sample["input_origin"] = input
        sample["input_norm"] = input_norm
        sample["input"] = np.expand_dims(np.array(input_norm, dtype=np.float32), axis=0)
        return sample
    

class CompressLgeSegDataset(torch.utils.data.Dataset) :
    def __init__(
        self,
        paths=None,
        compress_data_mu=None,
        compress_data_sig=None,
        processing=None,
        ):

        self.paths = paths
        self.comp_data_mu = np.array(compress_data_mu, dtype=np.float32)
        self.comp_data_sig = np.array(compress_data_sig, dtype=np.float32)
        self.processing = processing

        with open(self.processing.path_metrics, 'rb') as file:
            self.D_metrics = pickle.load(file)

        self.slice_counts = []
        self.slice_indices = []
        for path_dcm in self.paths["dcm.pkl"]:
            pat_name = path_dcm.split("/")[-2]

            idx_metrics_patient = self.D_metrics["pat_name"].index(pat_name)
            slices = self.D_metrics["index_slices"][idx_metrics_patient]
            self.slice_counts.append(len(slices))
            self.slice_indices.append(slices)
        self.cumulative_slices = np.cumsum([0] + self.slice_counts)  # Include 0 for easier indexing

    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        sample = {}

        # Find volume_idx using bisect for efficiency
        volume_idx = bisect_right(self.cumulative_slices, idx) - 1
        slice_idx  = idx - self.cumulative_slices[volume_idx]

        path_dcm = self.paths["dcm.pkl"][volume_idx]
        path_roi = self.paths["roi.pkl"][volume_idx]

        with open(path_dcm, 'rb') as file:
            dcm = pickle.load(file)
        with open(path_roi, 'rb') as file:
            roi = pickle.load(file)

        sample["pat_name"] = path_dcm.split("/")[-2]
        sample["seq_name"] = dcm.type_seq

        D_metric_patient = {}
        idx_metrics_patient = self.D_metrics["pat_name"].index(sample["pat_name"])
        slice_idx_in_list   = self.D_metrics["index_slices"][idx_metrics_patient][slice_idx]

        for key, val in self.D_metrics.items() :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                D_metric_patient[key] = val[idx_metrics_patient][slice_idx]
            else:
                D_metric_patient[key] = val[idx_metrics_patient]
        sample["metrics"] = D_metric_patient

        # # dict_keys(['pat_name', 'index_slices', 'z_vals', 'transmurality', 
        # # 'endo_surface_length', 'infarct_size_2D', 'angle_junction', 
        # # 'infarct_location', 'inversion_time', 'month', 'sequence_type', 
        # # 'age', 'sex', 'lv_mass', 'lv_edv', 'ejection_fraction', 'thrombus'])


        # Data dcm
        input_dcm = dcm.dataResampled[:,:,slice_idx_in_list]
        min_max = [np.min(input_dcm), np.max(input_dcm)]
        input_dcm_norm = (input_dcm - min_max[0])/(min_max[1] - min_max[0])

        # Data roi
        MI_and_nonMI = roi.segmentsResampled[slice_idx_in_list]['MI'] + roi.segmentsResampled[slice_idx_in_list]['non-MI']
        input_roi = np.array(MI_and_nonMI, dtype=np.float32)
        input_roi_norm = np.copy(input_roi)
        
        sample["input_origin_mod1"] = input_dcm
        sample["input_norm_mod1"] = input_dcm_norm
        sample["input_mod1"] = np.expand_dims(np.array(input_dcm_norm, dtype=np.float32), axis=0)

        sample["input_origin_mod2"] = input_roi
        sample["input_norm_mod2"] = input_roi_norm
        sample["input_mod2"] = np.expand_dims(np.array(input_roi_norm, dtype=np.float32), axis=0)

        sample["input_mu"]  = self.comp_data_mu[idx]
        sample["input_sig"]  = self.comp_data_sig[idx]
        return sample


class CompressLgeSegCondDataset(torch.utils.data.Dataset) :
    def __init__(
        self,
        paths=None,
        compress_data_mu=None,
        compress_data_sig=None,
        cond_data=None,
        processing=None,
        ):

        self.paths = paths
        self.comp_data_mu = np.array(compress_data_mu, dtype=np.float32)
        self.comp_data_sig = np.array(compress_data_sig, dtype=np.float32)
        self.cond_data = np.array(cond_data, dtype=np.float32)
        self.processing = processing

        with open(self.processing.path_metrics, 'rb') as file:
            self.D_metrics = pickle.load(file)

        self.slice_counts = []
        self.slice_indices = []
        for path_dcm in self.paths["dcm.pkl"]:
            pat_name = path_dcm.split("/")[-2]

            idx_metrics_patient = self.D_metrics["pat_name"].index(pat_name)
            slices = self.D_metrics["index_slices"][idx_metrics_patient]
            self.slice_counts.append(len(slices))
            self.slice_indices.append(slices)
        self.cumulative_slices = np.cumsum([0] + self.slice_counts)  # Include 0 for easier indexing

    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        sample = {}

        # Find volume_idx using bisect for efficiency
        volume_idx = bisect_right(self.cumulative_slices, idx) - 1
        slice_idx  = idx - self.cumulative_slices[volume_idx]

        path_dcm = self.paths["dcm.pkl"][volume_idx]
        path_roi = self.paths["roi.pkl"][volume_idx]

        with open(path_dcm, 'rb') as file:
            dcm = pickle.load(file)
        with open(path_roi, 'rb') as file:
            roi = pickle.load(file)

        sample["pat_name"] = path_dcm.split("/")[-2]
        sample["seq_name"] = dcm.type_seq

        D_metric_patient = {}
        idx_metrics_patient = self.D_metrics["pat_name"].index(sample["pat_name"])
        slice_idx_in_list   = self.D_metrics["index_slices"][idx_metrics_patient][slice_idx]

        for key, val in self.D_metrics.items() :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                D_metric_patient[key] = val[idx_metrics_patient][slice_idx]
            else:
                D_metric_patient[key] = val[idx_metrics_patient]
        sample["metrics"] = D_metric_patient

        # # dict_keys(['pat_name', 'index_slices', 'z_vals', 'transmurality', 
        # # 'endo_surface_length', 'infarct_size_2D', 'angle_junction', 
        # # 'infarct_location', 'inversion_time', 'month', 'sequence_type', 
        # # 'age', 'sex', 'lv_mass', 'lv_edv', 'ejection_fraction', 'thrombus'])

        # Data dcm
        input_dcm = dcm.dataResampled[:,:,slice_idx_in_list]
        min_max = [np.min(input_dcm), np.max(input_dcm)]
        input_dcm_norm = (input_dcm - min_max[0])/(min_max[1] - min_max[0])

        # Data roi
        MI_and_nonMI = roi.segmentsResampled[slice_idx_in_list]['MI'] + roi.segmentsResampled[slice_idx_in_list]['non-MI']
        input_roi = np.array(MI_and_nonMI, dtype=np.float32)
        input_roi_norm = np.copy(input_roi)
        
        sample["input_origin_mod1"] = input_dcm
        sample["input_norm_mod1"] = input_dcm_norm
        sample["input_mod1"] = np.expand_dims(np.array(input_dcm_norm, dtype=np.float32), axis=0)

        sample["input_origin_mod2"] = input_roi
        sample["input_norm_mod2"] = input_roi_norm
        sample["input_mod2"] = np.expand_dims(np.array(input_roi_norm, dtype=np.float32), axis=0)

        sample["input_mu"]   = self.comp_data_mu[idx]
        sample["input_sig"]  = self.comp_data_sig[idx]
        sample["input_cond"] = self.cond_data[idx]
        return sample


class CompressLgeSegCond_Scalars_Dataset(torch.utils.data.Dataset) :
    def __init__(
        self,
        paths=None,
        compress_data_mu=None,
        compress_data_sig=None,
        cond_data=None,
        processing=None,
        ):

        self.paths = paths
        self.comp_data_mu = np.array(compress_data_mu, dtype=np.float32)
        self.comp_data_sig = np.array(compress_data_sig, dtype=np.float32)
        self.cond_data = cond_data
        self.processing = processing

        with open(self.processing.path_metrics, 'rb') as file:
            self.D_metrics = pickle.load(file)

        self.slice_counts = []
        self.slice_indices = []
        for path_dcm in self.paths["dcm.pkl"]:
            pat_name = path_dcm.split("/")[-2]

            idx_metrics_patient = self.D_metrics["pat_name"].index(pat_name)
            slices = self.D_metrics["index_slices"][idx_metrics_patient]
            self.slice_counts.append(len(slices))
            self.slice_indices.append(slices)
        self.cumulative_slices = np.cumsum([0] + self.slice_counts)  # Include 0 for easier indexing      

    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        sample = {}

        # Find volume_idx using bisect for efficiency
        volume_idx = bisect_right(self.cumulative_slices, idx) - 1
        slice_idx  = idx - self.cumulative_slices[volume_idx]

        path_dcm = self.paths["dcm.pkl"][volume_idx]
        path_roi = self.paths["roi.pkl"][volume_idx]

        with open(path_dcm, 'rb') as file:
            dcm = pickle.load(file)
        with open(path_roi, 'rb') as file:
            roi = pickle.load(file)

        sample["pat_name"] = path_dcm.split("/")[-2]
        sample["seq_name"] = dcm.type_seq

        D_metric_patient = {}
        idx_metrics_patient = self.D_metrics["pat_name"].index(sample["pat_name"])
        slice_idx_in_list   = self.D_metrics["index_slices"][idx_metrics_patient][slice_idx]

        for key, val in self.D_metrics.items() :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                D_metric_patient[key] = val[idx_metrics_patient][slice_idx]
            else:
                D_metric_patient[key] = val[idx_metrics_patient]
        sample["metrics"] = D_metric_patient

        # # dict_keys(['pat_name', 'index_slices', 'z_vals', 'transmurality', 
        # # 'endo_surface_length', 'infarct_size_2D', 'angle_junction', 
        # # 'infarct_location', 'inversion_time', 'month', 'sequence_type', 
        # # 'age', 'sex', 'lv_mass', 'lv_edv', 'ejection_fraction', 'thrombus'])

        # Data dcm
        input_dcm = dcm.dataResampled[:,:,slice_idx_in_list]
        min_max = [np.min(input_dcm), np.max(input_dcm)]
        input_dcm_norm = (input_dcm - min_max[0])/(min_max[1] - min_max[0])

        # Data roi
        MI_and_nonMI = roi.segmentsResampled[slice_idx_in_list]['MI'] + roi.segmentsResampled[slice_idx_in_list]['non-MI']
        input_roi = np.array(MI_and_nonMI, dtype=np.float32)
        input_roi_norm = np.copy(input_roi)
        
        sample["input_origin_mod1"] = input_dcm
        sample["input_norm_mod1"] = input_dcm_norm
        sample["input_mod1"] = np.expand_dims(np.array(input_dcm_norm, dtype=np.float32), axis=0)

        sample["input_origin_mod2"] = input_roi
        sample["input_norm_mod2"] = input_roi_norm
        sample["input_mod2"] = np.expand_dims(np.array(input_roi_norm, dtype=np.float32), axis=0)

        sample["input_mu"]   = self.comp_data_mu[idx]
        sample["input_sig"]  = self.comp_data_sig[idx]

        L_cond = []
        for key in self.cond_data.keys() :
            if key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                L_cond.append(self.cond_data[key][volume_idx][int(slice_idx)])
            else:
                L_cond.append(self.cond_data[key][volume_idx])
        sample["input_cond"] = np.array(L_cond, dtype=np.float32)
        return sample


class CompressLgeSeg_CNCond_Dataset(torch.utils.data.Dataset) :
    def __init__(
        self,
        paths=None,
        compress_data_mu=None,
        compress_data_sig=None,
        compress_segm_mu=None,
        processing=None,
        ):

        self.paths = paths
        self.comp_data_mu  = np.array(compress_data_mu, dtype=np.float32)
        self.comp_data_sig = np.array(compress_data_sig, dtype=np.float32)
        self.comp_segm_mu  = np.array(compress_segm_mu, dtype=np.float32)
        self.processing = processing

        with open(self.processing.path_metrics, 'rb') as file:
            self.D_metrics = pickle.load(file)

        self.slice_counts = []
        self.slice_indices = []
        for path_dcm in self.paths["dcm.pkl"]:
            pat_name = path_dcm.split("/")[-2]

            idx_metrics_patient = self.D_metrics["pat_name"].index(pat_name)
            slices = self.D_metrics["index_slices"][idx_metrics_patient]
            self.slice_counts.append(len(slices))
            self.slice_indices.append(slices)
        self.cumulative_slices = np.cumsum([0] + self.slice_counts)  # Include 0 for easier indexing

    def __len__(self):
        return self.cumulative_slices[-1]
    
    def __getitem__(self, idx):
        sample = {}

        # Find volume_idx using bisect for efficiency
        volume_idx = bisect_right(self.cumulative_slices, idx) - 1
        slice_idx  = idx - self.cumulative_slices[volume_idx]

        path_dcm = self.paths["dcm.pkl"][volume_idx]
        path_roi = self.paths["roi.pkl"][volume_idx]

        with open(path_dcm, 'rb') as file:
            dcm = pickle.load(file)
        with open(path_roi, 'rb') as file:
            roi = pickle.load(file)

        sample["pat_name"] = path_dcm.split("/")[-2]
        sample["seq_name"] = dcm.type_seq

        D_metric_patient = {}
        idx_metrics_patient = self.D_metrics["pat_name"].index(sample["pat_name"])
        slice_idx_in_list   = self.D_metrics["index_slices"][idx_metrics_patient][slice_idx]

        for key, val in self.D_metrics.items() :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                D_metric_patient[key] = val[idx_metrics_patient][slice_idx]
            else:
                D_metric_patient[key] = val[idx_metrics_patient]
        sample["metrics"] = D_metric_patient

        # # dict_keys(['pat_name', 'index_slices', 'z_vals', 'transmurality', 
        # # 'endo_surface_length', 'infarct_size_2D', 'angle_junction', 
        # # 'infarct_location', 'inversion_time', 'month', 'sequence_type', 
        # # 'age', 'sex', 'lv_mass', 'lv_edv', 'ejection_fraction', 'thrombus'])


        # Data dcm
        input_dcm = dcm.dataResampled[:,:,slice_idx_in_list]
        min_max = [np.min(input_dcm), np.max(input_dcm)]
        input_dcm_norm = (input_dcm - min_max[0])/(min_max[1] - min_max[0])

        # Data roi
        MI_and_nonMI = roi.segmentsResampled[slice_idx_in_list]['MI'] + roi.segmentsResampled[slice_idx_in_list]['non-MI']
        input_roi = np.array(MI_and_nonMI, dtype=np.float32)
        input_roi_norm = np.copy(input_roi)
        
        sample["input_origin_mod1"] = input_dcm
        sample["input_norm_mod1"] = input_dcm_norm
        sample["input_mod1"] = np.expand_dims(np.array(input_dcm_norm, dtype=np.float32), axis=0)

        sample["input_origin_mod2"] = input_roi
        sample["input_norm_mod2"] = input_roi_norm
        sample["input_mod2"] = np.expand_dims(np.array(input_roi_norm, dtype=np.float32), axis=0)

        sample["input_mu"]  = self.comp_data_mu[idx]
        sample["input_sig"] = self.comp_data_sig[idx]
        sample["segm_mu"]   = self.comp_segm_mu[idx]
        return sample
    




