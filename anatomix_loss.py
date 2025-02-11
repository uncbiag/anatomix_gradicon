import icon_registration as icon
from icon_registration import networks
import torch

import anatomix

class AnatomixLossOuter(icon.RegistrationModule):
    def __init__(self, regis_net):
        super().__init__()
        self.regis_net = regis_net
        self.anatomix = anatomix.load_model()

    def forward(self, image_A, image_B):
        import pdb
#        pdb.set_trace()
        image_A_features = anatomix.extract_features(image_A, self.anatomix)
        image_B_features = anatomix.extract_features(image_B, self.anatomix)

        image_A = torch.cat([image_A, image_A_features], dim=1)
        image_B = torch.cat([image_B, image_B_features], dim=1)

        result = self.regis_net(image_A, image_B)

        self.warped_image_A = self.regis_net.warped_image_A
        self.warped_image_B = self.regis_net.warped_image_B
        self.phi_AB_vectorfield = self.regis_net.phi_AB_vectorfield

        return result

class AnatomixSSD(icon.losses.SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."

        import pdb
        #pdb.set_trace()
        return torch.mean((image_A[:, 1:] - image_B[:, 1:]) ** 2)

class StripAnatomix(icon.RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, moving, fixed):
        return self.net(moving[:, :1], fixed[:, :1])

input_shape = [1, 1, 175, 175, 175]

def make_network():
  inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

  for _ in range(3):
       inner_net = icon.TwoStepRegistration(
           icon.DownsampleRegistration(inner_net, dimension=3),
           icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
       )
  inner_net = icon.TwoStepRegistration(
           inner_net,
           icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
       )

  net = AnatomixLossOuter(
          icon.losses.GradientICONSparse(StripAnatomix(inner_net), AnatomixSSD(), lmbda=6.5))
  net.assign_identity_map(input_shape)
  return net
