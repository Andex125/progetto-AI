import torch
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import torch_spline_conv
import pyg_lib

print("Torch:", torch.version)
print("CUDA disponibile:", torch.cuda.is_available())
print("PyTorch Geometric:", torch_geometric.__version__)
print("torch_scatter:", torch_scatter.version)
print("torch_sparse:", torch_sparse.version)
print("torch_cluster:", torch_cluster.version)
print("torch_spline_conv:", torch_spline_conv.version)
print("pyg_lib:", pyg_lib.version)