# Adversarial Training and Visualization on MNIST


## Results

### Learning Curves

Epsilons in linf (l2) training is 0.3 (1.5). 

<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> <strong>Standard Training</strong> </td>
			<td width="27%" align="center"> <strong>l_inf Training</strong> </td>
			<td width="27%" align="center"> <strong>l_2 Training</strong></td>
		</tr>
<tr>
			<td width="27%" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_learning_curve_std.jpg"> </td>
			<td width="27%" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_learning_curve_linf.jpg"> </td>
			<td width="27%" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_learning_curve_l2.jpg"> </td>
		</tr>
	</tbody>
</table>


### Visualization of Gradient with Respect to Input

![visualization](https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_grad_default.jpg)

### The Adversarial Example with large epsilon

The maximum epsilon is set to 4 (l2 norm) in this part.

![large](https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_large_l2_.jpg)


## Requirements:
```
python >= 3.5
torch == 1.0
torchvision == 0.2.1
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
python main.py --data_root [data directory] -e 0.3 -p 'linf' --adv_train
```

l2 training: <br/>

```
python main.py --data_root [data directory] -e 1.5 -p 'l2' --adv_train
```

### Visualization

visualize gradient to input: <br/>

```
python visualize.py --load_checkpoint [your_model.pth]
```

visualize adversarial examples with larger epsilon <br/>

```
python visualize_attack.py --load_checkpoint [your_model.pth]
```


## Training Time

Standard training: 0.64 s / 100 iterations <br/>
Adversarial training: 16 s / 100 iterations <br/> <br/>

where the batch size is 64 and train on NVIDIA GeForce GTX 1080.
