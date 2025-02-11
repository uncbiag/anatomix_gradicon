
# train.py

import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch

from anatomix_loss import input_shape, make_network

BATCH_SIZE = 3
GPUS = 4

def make_batch():
    image = torch.cat([random.choice(images) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image.float()


if __name__ == "__main__":
    footsteps.initialize()
    images = torch.load(
        "/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/stretched_traintensor/train_imgs_tensor.trch", weights_only=True
    )
    #hg
    net = make_network()

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=net)
