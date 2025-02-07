# model.py

import icon_registration as icon
from icon_registration import networks

input_shape = [1, 1, 175, 175, 175]

def make_network():
  inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

  for _ in range(2):
       inner_net = icon.TwoStepRegistration(
           icon.DownsampleRegistration(inner_net, dimension=3),
           icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
       )
  inner_net = icon.TwoStepRegistration(
           inner_net,
           icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
       )

  net = icon.GradientICON(inner_net, icon.LNCC(sigma=4), lmbda=1.5)
  net.assign_identity_map(input_shape)
  return net
