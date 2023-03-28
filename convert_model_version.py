import torch

path = 'path/to/SOLIDER/log/lup/swin_tiny/checkpoint.pth'
state_dict = torch.load(path)
state_dict = state_dict["teacher"]
torch.save(state_dict, path.replace('.pth','_new.pth'), _use_new_zipfile_serialization=False)
