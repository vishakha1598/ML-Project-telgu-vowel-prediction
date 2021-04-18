# ML-Project-telgu-vowel-prediction

dataset link : "https://www.kaggle.com/syamkakarla/telugu-6-vowel-dataset"

Algorithm Used : ResNet50
_**Architecture of ResNet50**_
![image](https://user-images.githubusercontent.com/64003365/115149273-f2851c80-a080-11eb-9eba-ee829a0a1bf0.png)

**ResNet50 function
tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
)**


Arguments

**include_top**: whether to include the fully-connected layer at the top of the network.
**weights**: one of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
**input_tensor**: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
**input_shape**: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
**pooling**: Optional pooling mode for feature extraction when include_top is False.
None means that the output of the model will be the 4D tensor output of the last convolutional block.
avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
max means that global max pooling will be applied.
**classes**: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
