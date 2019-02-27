# Adversarial Training and Visualization

The repo is the PyTorch-1.0 implementation for the adversarial training on MNIST/CIFAR-10. And I also reproduce part of the visualization results in [1]. <br/><br/>

**Note**: Not an official implementation.

## Adversarial Training

<table align="center">
    <tbody> 
    <tr> 
        <th colspan="2"> Objective Function  </th>
    </tr>
    <tr>        
        <td width="50%" align="center"> Standard Training </td>
        <td width="50%" align="center"> Adversarial Training </td>
    </tr>
    <tr>
        <td width="50%" align="center"> <img src="https://latex.codecogs.com/gif.latex?\min&space;\textrm{E}_{(x,&space;y)&space;\in&space;Dataset}[L(x,&space;y;&space;\theta))]" title="\min \textrm{E}_{(x, y) \in Dataset}[L(x, y; \theta))]"> </td>
        <td width="50%" align="center"> <img src="https://latex.codecogs.com/gif.latex?\min&space;\textrm{E}_{(x,&space;y)&space;\in&space;Dataset}[\max_{{\left&space;\|&space;\delta&space;\right&space;\|}_p&space;<&space;\epsilon}&space;L(x&plus;\delta,&space;y;&space;\theta))]" title="\min \textrm{E}_{(x, y) \in Dataset}[\max_{{\left \| \delta \right \|}_p < \epsilon} L(x+\delta, y; \theta))]"> </td>
    </tr>
    </tbody>
</table>

where p in the table is usually 2 or inf. <br/><br/>

The objective of standard and adversarial training is fundamentally different. In standard training, the classifier minimize the loss computed from the original training data, while in adversarial training, it trains with the worst-case around the original data.

## Visualization

In [1], the authors discover that the features learned by the robustness classifier are more human-perceivable. Related results are shown in mnist/cifar-10 folder.

## Implementation

Part of the codes in this repo are borrowed/modified from [2], [3], [4] and [5].

## References:

[1] D. Tsipras, S. Santurkar, L. Engstrom, A. Turner, A. Madry. *Robustness May Be at Odds with Accuracy*, https://arxiv.org/abs/1805.12152

[2] https://github.com/MadryLab/mnist_challenge

[3] https://github.com/MadryLab/cifar10_challenge

[4] https://github.com/xternalz/WideResNet-pytorch

[5] https://github.com/utkuozbulak/pytorch-cnn-visualizations


## Contact 
Yi-Lin Sung, corumlouis123@gmail.com
