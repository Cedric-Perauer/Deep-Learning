import torch
import struct
from utils.torch_utils import select_device
import os 

a = os.listdir()
wts = []
for f in a : 
 if f[-2:] == "pt":
     wts.append(f)

# Initialize
device = select_device('cpu')
# Load model

for f in wts : 
   model = torch.load(f, map_location=device)['model'].float()  # load to FP32
   model.to(device).eval()

   f = open(f[:-3]+".wts", 'w')
   f.write('{}\n'.format(len(model.state_dict().keys())))
   for k, v in model.state_dict().items():
     vr = v.reshape(-1).cpu().numpy()
     f.write('{} {} '.format(k, len(vr)))
     for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
     f.write('\n')
