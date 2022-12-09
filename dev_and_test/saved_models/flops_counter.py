"""
import torch
from thop import profile

model = torch.load("model.pt")
ins = {"obs":torch.randn(4096, 84, 84, 3), "state_ins":torch.randn(4096,)}
macs, params = profile(model, inputs=ins)
"""
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = torch.load("model.pt")
  macs, params = get_model_complexity_info(net, (3, 84, 84), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))