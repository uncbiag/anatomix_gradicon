# model.py

import unigradicon

input_shape = [1, 1, 175, 175, 175]

"""
image format: concatenate one-hot segmentations with preprocessed images along channel dimension, 
to get an image with shape B x (1 + num_segs) x H x W x D
"""


class SegmentationSSD(SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, image_A, image_B):
        assert image_A.shape == image_B.shape, "The shape of image_A and image_B sould be the same."
        return torch.mean((image_A[1:] - image_B[1:]) ** 2)

class StripSegmentations(icon.RegistrationModule):
    def __init__(self, net):
        self.net = net
    def forward(self, moving, fixed):
        return self.net(moving[:, :1], fixed[:, :1])

def make_network():

    multigradicon = unigradicon.get_multigradicon(loss_fn=SegmentationSSD())

    multigradicon.regis_net = StripSegmentations(multigradicon.regis_net) 

    multigradicon.assign_identity_map(multigradicon.identity_map.shape)
    multigradicon.to(config.device)
    return multigradicon
