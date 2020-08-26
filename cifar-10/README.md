# Adversarial Training and Visualization on CIFAR-10


## Update
* (2020/8/27)
1. To match the implementation of [madry_cifar10](https://github.com/MadryLab/cifar10_challenge), we update the default learning rate to `0.1`, the activation function of model to `LeakyReLU(0.1)`, and the optimizer change to `torch.SGD`.
2. Add new experiment in *Quantitative Results*, which match the results in [madry_cifar10](https://github.com/MadryLab/cifar10_challenge).
3. Add checkpoints for the updated model and delete the old ones.
4. Update codes structure. Pull `main.py`, `visualize.py` and `visualize_attack.py` out of `src` folder.
* (2019/12/14) 
1. Add `madry_model.py`, which contains the same model used in [madry_cifar10](https://github.com/MadryLab/cifar10_challenge), in `src/model`. 
2. Add `count_parameters(model)` in `model.py` and `madry_model.py`, and it can compute the number of all the trainable parameters.
3. Flag `use_pseudo_label`, which determine whether to use model's prediction as the target, is added in to `trainer.test()`, and the default value is `False`.
4. Update the "Experiments" and the "Execution" part in this Readme. 
5. Checkpoints of adversarial training in cifar-10 are provided.
* (2019/4/18) Change the default alpha from 2 to 2/255, and update the results.

## Results

Note that the experiments only conduct 1 time.

### Learning Curves

Epsilon in linf (l2) training is 0.0157 (0.314). [0.0157=4/255, 0.314=80/255]

<table border=0 width="50px" >
    <tbody> 
    <tr>    
        <th colspan="2" align="center"> <strong>Standard Training</strong> </th>
        <th colspan="2" align="center"> <strong>l_inf Training</strong> </th>
        <th colspan="2" align="center"> <strong>l_2 Training</strong></th>
    </tr>
    <tr>
        <th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_learning_curve_std.jpg"> </th>
        <th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_learning_curve_linf.jpg"> </th>
        <th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_learning_curve_l2.jpg"> </th>
    </tr>
    <tr>
        <th colspan="1" align="center"> <strong>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Robustness Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Robustness Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Robustness Accuracy</strong> <br/> (train/test) </th>
    </tr>
    <tr>
        <th colspan="1" align="center"> 92.19/87.14 </th>
        <th colspan="1" align="center"> 0.00/7.85 </th>
        <th colspan="1" align="center"> 79.69/78.09 </th>
        <th colspan="1" align="center"> 61.72/63.8 </th>
        <th colspan="1" align="center"> 89.84/85.39 </th>
        <th colspan="1" align="center"> 76.56/77.76 </th>
    </tr>
    <tr>
        <th colspan="1" align="center"> <strong>Madry's Model <br/>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Madry's Model <br/>Robustness Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Madry's Model <br/>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Madry's Model <br/>Robustness Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Madry's Model <br/>Standard Accuracy</strong> <br/> (train/test) </th>
        <th colspan="1" align="center"> <strong>Madry's Model <br/>Robustness Accuracy</strong> <br/> (train/test) </th>
    </tr>
    <tr>
        <th colspan="1" align="center"> - </th>
        <th colspan="1" align="center"> - </th>
        <th colspan="1" align="center"> -/79.22 </th>
        <th colspan="1" align="center"> -/55.97 </th>
        <th colspan="1" align="center"> -/85.81 </th>
        <th colspan="1" align="center"> -/71.87 </th>
    </tr>
    </tbody>
</table>

(Only refer to those results which are not Madry's Model) Note that in testing mode, the target label used in creating the adversarial example is the most confident prediction of the model, not the ground truth. Therefore, sometimes the testing robustness is higher than training robustness, when the prediction is wrong at first. <br/>

Learning rate is manually changed during training: <br/>

* `0.1` in iteration `[0, 40000]`
* `0.01` in iteration `[40000, 60000]`
* `0.001` in iteration `[60000, 76000]`

the policy is followed https://github.com/MadryLab/cifar10_challenge.


### Quantitative Results

* Defense model, standard accuracy = 86.66% (linf, epsilon=8/255) (train on PGD attack with 7 steps of size 2)

|   Attack                                      | Robust Test Accuracy  |
|   :---:                                       |  :---:                |
|   PGD with 10 steps of size 2 (cross-entropy) |    48.37%             |
|   PGD with 20 steps of size 1 (cross-entropy) |    48.04%             |

### Visualization of Gradient with Respect to Input

![visualization](https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_grad_default.jpg)

### The Adversarial Example with large epsilon

The maximum epsilon is set to 4.7 (l2 norm) in this part.

![large](https://github.com/louis2889184/adversarial_training/blob/master/cifar-10/img/cifar_large_l2_default.jpg)


## Requirements:
```
python >= 3.5
torch >= 1.0
torchvision >= 0.2.1
numpy >= 1.16.1
matplotlib >= 3.0.2
```

## Execution

### Training

Standard training: <br/>

```
python main.py --data_root [data directory]
```

linf training: <br/>

```
python main.py --data_root [data directory] -e 0.0157 -p 'linf' --adv_train --affix 'linf'
```

l2 training: <br/>

```
python main.py --data_root [data directory] -e 0.314 -p 'l2' --adv_train --affix 'l2'
```

### Testing

change the setting if you want to do linf testing.
```
python main.py --todo test --data_root [data directory] -e 0.314 -p 'l2' --load_checkpoint [your_model.pth]
```

### Visualization

change the setting in `visualize.py` `visualize_attack.py` and if you want to do linf visualization.

visualize gradient to input: <br/>

```
python visualize.py --load_checkpoint [your_model.pth]
```

visualize adversarial examples with larger epsilon <br/>

```
python visualize_attack.py --load_checkpoint [your_model.pth]
```

## Checkpoints
### linf
* epsilon=8/255, train on PGD attack with 7 steps of size 2: [checkpoint](https://drive.google.com/file/d/1-3AfpkLvPje5poY9ZettY05N8kZgFRAV/view?usp=sharing) <br/>

## Training Time

Standard training: 78 s / 100 iterations <br/>
Adversarial training: 784 s / 100 iterations <br/> <br/>

where the batch size is 128 and train on NVIDIA GeForce GTX 1080.
