import h5py, numpy as np

fpath = "/home/yuttokb/data/dataset_1_pions_2.hdf5"          # ↰ ajusta la ruta
with h5py.File(fpath, "r") as f:
    dset = f["showers"]
    print("Forma completa  :", dset.shape)       # p. ej. (120 800, 533)
    print("Un evento       :", dset[0].shape)    # p. ej. (533,)

