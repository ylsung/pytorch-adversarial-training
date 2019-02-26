
import os
import torch
import torchvision as tv
import numpy as np

from torch.utils.data import DataLoader

from utils import makedirs, tensor2cuda, load_model
from argument import parser
from visualization import VanillaBackprop
from attack import FastGradientSignUntargeted
from model import Model

import matplotlib.pyplot as plt 

img_folder = '../img'
makedirs(img_folder)

args = parser()


te_dataset = tv.datasets.MNIST(args.data_root, 
                               train=False, 
                               transform=tv.transforms.ToTensor(), 
                               download=True)

te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


for data, label in te_loader:

    data, label = tensor2cuda(data), tensor2cuda(label)


    break

types = ['Original', 'Standard', r'$l_{\infty}$-trained', r'$l_2$-trained']


model_checkpoints = ['../checkpoint/mnist_std_train/checkpoint_56000.pth',
                     '../checkpoint/mnist_adv_train/checkpoint_56000.pth', 
                     '../checkpoint/mnist_l2_adv/checkpoint_56000.pth']

adv_list = []
pred_list = []

max_epsilon = 0.8

perturbation_type = 'linf'

with torch.no_grad():
    for checkpoint  in model_checkpoints:

        model = Model(i_c=1, n_c=10)

        load_model(model, checkpoint)

        if torch.cuda.is_available():
            model.cuda()

        attack = FastGradientSignUntargeted(model, 
                                            max_epsilon, 
                                            args.alpha, 
                                            min_val=0, 
                                            max_val=1, 
                                            max_iters=args.k, 
                                            _type=perturbation_type)

       
        adv_data = attack.perturb(data, label, 'mean', False)

        output = model(adv_data, _eval=True)
        pred = torch.max(output, dim=1)[1]
        adv_list.append(adv_data.cpu().numpy().squeeze())  # (N, 28, 28)
        pred_list.append(pred.cpu().numpy())

data = data.cpu().numpy().squeeze()  # (N, 28, 28)
data *= 255.0
label = label.cpu().numpy()

adv_list.insert(0, data)

pred_list.insert(0, label)

out_num = 5

fig, _axs = plt.subplots(nrows=len(adv_list), ncols=out_num)

axs = _axs

cmap = 'gray'
for j, _type in enumerate(types):
    axs[j, 0].set_ylabel(_type)

    for i in range(out_num):
        axs[j, i].set_xlabel('%d' % pred_list[j][i])
        axs[j, i].imshow(adv_list[j][i], cmap=cmap)

        axs[j, i].get_xaxis().set_ticks([])
        axs[j, i].get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'mnist_large_%s_%s.jpg' % (perturbation_type, args.affix)))