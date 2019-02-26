# Adversarial Training and Visualization on MNIST

## Learning Curves


<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> <strong>Standard Training</strong> </td>
			<td width="27%" align="center"> <strong>linf Training</strong> </td>
			<td width="27%" align="center"> <strong>l2 Training</strong></td>
		</tr>
<tr>
			<td width="27%" align="center"> <img src="/img/mnist_learning_curve_std.jpg"> </td>
			<td width="27%" align="center"> <img src="/img/mnist_learning_curve_linf.jpg"> </td>
			<td width="27%" align="center"> <img src="/img/mnist_learning_curve_l2.jpg"> </td>
		</tr>
	</tbody>
</table>




## Requirements:
```
torch >= 0.4.0
torchvision >= 0.1.9
numpy >= 1.13.0
matplotlib >= 1.5
PIL >= 1.1.7
```


## References:

[1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. *Striving for Simplicity: The All Convolutional Net*, https://arxiv.org/abs/1412.6806


