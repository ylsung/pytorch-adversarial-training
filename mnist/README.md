# Adversarial Training and Visualization on MNIST


## Results

### Learning Curves

Epsilon in linf (l2) training is 0.3 (1.5). 

<table border=0 width="50px" >
	<tbody> 
    <tr>	
    	<th colspan="2" align="center"> <strong>Standard Training</strong> </th>
		<th colspan="2" align="center"> <strong>l_inf Training</strong> </th>
		<th colspan="2" align="center"> <strong>l_2 Training</strong></th>
	</tr>
	<tr>
		<th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_learning_curve_std.jpg"> </th>
		<th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_learning_curve_linf.jpg"> </th>
		<th colspan="2" align="center"> <img src="https://github.com/louis2889184/adversarial_training/blob/master/mnist/img/mnist_learning_curve_l2.jpg"> </th>
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
		<th colspan="1" align="center"> 100.00/99.32 </th>
		<th colspan="1" align="center"> 0.00/0.61 </th>
		<th colspan="1" align="center"> 100.00/98.96 </th>
		<th colspan="1" align="center"> 96.88/95.16 </th>
		<th colspan="1" align="center"> 100.00/99.41 </th>
		<th colspan="1" align="center"> 100.00/97.48 </th>
	</tr>
	</tbody>
</table>

Note that in testing mode, the target label used in creating the adversarial example is the most confident prediction of the model, not the ground truth. Therefore, sometimes the testing robustness is higher than training robustness, when the prediction is wrong at first.

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


## Training Time

Standard training: 0.64 s / 100 iterations <br/>
Adversarial training: 16 s / 100 iterations <br/> <br/>

where the batch size is 64 and train on NVIDIA GeForce GTX 1080.
