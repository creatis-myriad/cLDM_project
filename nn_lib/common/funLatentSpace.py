import numpy as np



def GetSigmaPoints(emb, n_comp, d1, d2):
    d_mean = np.array([np.mean(emb[:,i]) for i in range(n_comp)], dtype=np.float32)
    d_std  = np.array([np.std(emb[:,i]) for i in range(n_comp)], dtype=np.float32)
    points = np.array([d_mean for _ in range (10)], dtype=np.float32)
    coeff_sigma = np.array([-2,-1,0,1,2], dtype=np.float32)
    points[:len(coeff_sigma),d1] += coeff_sigma*d_std[d1]
    points[len(coeff_sigma):,d2] += coeff_sigma*d_std[d2]
    return points
