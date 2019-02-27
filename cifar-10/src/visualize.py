
import os
import torch
import torchvision as tv
import numpy as np

from torch.utils.data import DataLoader

from utils import makedirs, tensor2cuda, load_model, LabelDict
from argument import parser
from visualization import VanillaBackprop
from model import WideResNet

import matplotlib.pyplot as plt 

img_folder = 'img'
makedirs(img_folder)
out_num = 5


args = parser()

label_dict = LabelDict(args.dataset)

te_dataset = tv.datasets.CIFAR10(args.data_root, 
                               train=False, 
                               transform=tv.transforms.ToTensor(), 
                               download=True)

te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


for data, label in te_loader:

    data, label = tensor2cuda(data), tensor2cuda(label)


    break


model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

load_model(model, args.load_checkpoint)

if torch.cuda.is_available():
    model.cuda()

VBP = VanillaBackprop(model)

grad = VBP.generate_gradients(data, label)

grad_flat = grad.view(grad.shape[0], -1)
mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3)
std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3)

mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
std = std.repeat(1, 1, data.shape[2], data.shape[3])

grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)

print(grad.min(), grad.max())

grad -= grad.min()

grad /= grad.max()

grad = grad.cpu().numpy().squeeze()  # (N, 28, 28)

grad *= 255.0

label = label.cpu().numpy()

data = data.cpu().numpy().squeeze()

data *= 255.0

out_list = [data, grad]

types = ['Original', 'Your Model']

fig, _axs = plt.subplots(nrows=len(out_list), ncols=out_num)

axs = _axs

for j, _type in enumerate(types):
    axs[j, 0].set_ylabel(_type)

    # if j == 0:
    #     cmap = 'gray'
    # else:
    #     cmap = 'seismic'

    for i in range(out_num):
        axs[j, i].set_xlabel('%s' % label_dict.label2class(label[i]))
        img = out_list[j][i]
        # print(img)
        img = np.transpose(img, (1, 2, 0))

        img = img.astype(np.uint8)
        axs[j, i].imshow(img)

        axs[j, i].get_xaxis().set_ticks([])
        axs[j, i].get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'cifar_grad_%s.jpg' % args.affix))

# types = ['Original', 'Standard', r'$l_{\infty}$-trained', r'$l_2$-trained']


# model_checkpoints = ['checkpoint/cifar-10_std/checkpoint_76000.pth',
#                      'checkpoint/cifar-10_linf/checkpoint_76000.pth', 
#                      'checkpoint/cifar-10_l2/checkpoint_76000.pth']


# out_list = []

# for checkpoint in model_checkpoints:

#     model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

#     load_model(model, checkpoint)

#     if torch.cuda.is_available():
#         model.cuda()

#     VBP = VanillaBackprop(model)

#     grad = VBP.generate_gradients(data, label)

#     grad_flat = grad.view(grad.shape[0], -1)
#     mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3)
#     std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3)

#     mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
#     std = std.repeat(1, 1, data.shape[2], data.shape[3])

#     grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)

#     print(grad.min(), grad.max())

#     grad -= grad.min()

#     grad /= grad.max()

#     grad = grad.cpu().numpy().squeeze()  # (N, 28, 28)

#     grad *= 255.0

#     out_list.append(grad)

# data = data.cpu().numpy().squeeze()  # (N, 28, 28)
# data *= 255.0
# label = label.cpu().numpy()

# out_list.insert(0, data)

# # normalize the grad
# # length = torch.norm(grad, dim=3)
# # length = torch.norm(length, dim=2)
# # length = length.unsqueeze(2).unsqueeze(2)
# # grad /= (length + 1e-5)

# out_num = 5

# fig, _axs = plt.subplots(nrows=len(out_list), ncols=out_num)

# axs = _axs


# for j, _type in enumerate(types):
#     axs[j, 0].set_ylabel(_type)

#     # if j == 0:
#     #     cmap = 'gray'
#     # else:
#     #     cmap = 'seismic'

#     for i in range(out_num):

#         data_id = i + 0

#         axs[j, i].set_xlabel('%s' % label_dict.label2class(label[data_id]))
        
#         img = out_list[j][data_id]
#         # print(img)
#         img = np.transpose(img, (1, 2, 0))

#         img = img.astype(np.uint8)
#         axs[j, i].imshow(img)

#         axs[j, i].get_xaxis().set_ticks([])
#         axs[j, i].get_yaxis().set_ticks([])

# plt.tight_layout()
# plt.savefig(os.path.join(img_folder, 'cifar_grad_%s.jpg' % args.affix))