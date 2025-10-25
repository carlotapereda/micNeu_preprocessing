#seaad check UMI and X layers

#import zarr, numpy as np


#root = zarr.open("/mnt/data/seaad_dlpfc/SEAAD_full_v2_streamed.zarr", mode="r")

#if "layers" in root:
#    print("Layer keys:", list(root["layers"].array_keys()))
#else:
#    print("No layers group in this Zarr.")



#root = zarr.open("/mnt/data/seaad_dlpfc/SEAAD_full_v2_streamed.zarr", mode="r")
#X = root["X"]

#print("Shape:", X.shape)
#print("Chunks:", X.chunks)
#print("dtype:", X.dtype)

# Read only a small chunk directly from disk
#block = X[0:2000, 0:50]
#print("min:", block.min(), "max:", block.max())
#print("unique (first row):", np.unique(block[0])[:20])


import zarr, numpy as np
root = zarr.open("/mnt/data/seaad_dlpfc/SEAAD_v2_XisUMIs.zarr")
X = root["X"]
print(np.unique(X[0, :50]))
