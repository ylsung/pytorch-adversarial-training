# Adversarial Training and Visualization on MNIST

## Learning Curves

<!-- 
<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> </td>
			<td width="27%" align="center"> <strong>Vanilla Backprop</strong> </td>
			<td width="27%" align="center"> </td>
		</tr>
<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/vanilla/snake_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/vanilla/dog_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/vanilla/spider_.gif"> </td>
		</tr>
	</tbody>
</table>


<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> </td>
			<td width="27%" align="center"> <strong>Guided Backprop</strong> </td>
			<td width="27%" align="center"> </td>
		</tr>
<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/gbp/snake_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/gbp/dog_.gif"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/gbp/spider_.gif"> </td>
		</tr>
	</tbody>
</table>


## General Information

Depending on the technique, the code uses pretrained **AlexNet** or **VGG** from the model zoo. Some of the code also assumes that the layers in the model are separated into two sections; **features**, which contains the convolutional layers and **classifier**, that contains the fully connected layer (after flatting out convolutions). If you want to port this code to use it on your model that does not have such separation, you just need to do some editing on parts where it calls *model.features* and *model.classifier*.

Every technique has its own python file (e.g. *gradcam.py*) which I hope will make things easier to understand. *misc_functions.py* contains functions like image processing and image recreation which is shared by the implemented techniques.

All images are pre-processed with mean and std of the ImageNet dataset before being fed to the model. None of the code uses GPU as these operations are quite fast for a single image (except for deep dream because of the example image that is used for it is huge). You can make use of gpu with very little effort. The example pictures below include numbers in the brackets after the description, like *Mastiff (243)*, this number represents the class id in the ImageNet dataset.

I tried to comment on the code as much as possible, if you have any issues understanding it or porting it, don't hesitate to send an email or create an issue.

Below, are some sample results for each operation.
 -->

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


