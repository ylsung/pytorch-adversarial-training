"""
this code is modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations

original author: Utku Ozbulak - github.com/utkuozbulak
"""

import sys
sys.path.append("..")

import torch

from src.utils import tensor2cuda, one_hot

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model

    def generate_gradients(self, input_image, target_class):
        # Put model in evaluation mode
        self.model.eval()

        x = input_image.clone()

        x.requires_grad = True

        with torch.enable_grad():
            # Forward
            model_output = self.model(x)
            # Zero grads
            self.model.zero_grad()
            
            grad_outputs = one_hot(target_class, model_output.shape[1])
            grad_outputs = tensor2cuda(grad_outputs)

            grad = torch.autograd.grad(model_output, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

            self.model.train()

        return grad
