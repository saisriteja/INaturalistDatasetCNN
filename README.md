# INaturalistDatasetCNN



## Dataset
INaturalist dataset have 10 classes, each class of size 1000 images. These are split to train, test and val(75,25,25). The images are resized and then fed to the nerual network which are pretrained models, these include 'Inception3','InceptionResNetV2', 'ResNet50', 'Xception' in experimentation.

Regularization Techniques used
1.  Data Agumentations
2.  Early Stopping
3.  Batch Normalization
4.  Learning Rate 


## Inferences
1.  Image net weights when set to true have performed because of better weight initialisation.
2.  InceptionResNetV2 and Xception have performed better than InceptionV3. ResNet50 was the least performing model.
3.  Data augmentation along with the batch size and dropout is negatively correlated. 
4.  The best validation accuracy that we have got so far is \textbf{\small 79.4 \%}79.4 % with InceptionResNetV2 model.
5.  Freezing 80 to 90 percent of the layers has performed better in terms of validation accuracy.
6.  The dense layer of size 128 has given good results than layers with sizes 512 and 256 implying that choosing a good dense layer size remains crucial.



I have used wandb for monitoring all the models and check which is a better one for experimentation and during production. 


![plotanalysis](https://user-images.githubusercontent.com/48018142/163723272-aea7167b-cd3b-43c6-9d83-28200a15e84f.JPG)
![parallel](https://user-images.githubusercontent.com/48018142/163723273-8d616d9f-19c1-4ca6-93b6-c277a809e469.JPG)






## Filter
The image below shows what a filter see in the convolutional network.
```

```

![truelabel](https://user-images.githubusercontent.com/48018142/163723275-8a14470a-1fc7-44f6-9ad2-8d7f33d5d62b.JPG =250x250)





## Gradient Checking



```

# This custom model has the 5th convolutional layer as its final layer
guided_backprop_model = tf.keras.models.Model(inputs = [model.inputs], outputs = [model.get_layer(index=-8).output])
# Here we choose only those layers that have an activation attribute
layer_dictionary = [layer for layer in guided_backprop_model.layers[1:] if hasattr(layer,'activation')]

# Define a custom gradient for the version of ReLU needed for guided backpropagation
@tf.custom_gradient
def guidedbackpropSelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

for l in layer_dictionary:
    # Change the ReLU activation to supress the negative gradients
    if l.activation == tf.keras.activations.selu:
        l.activation = guidedbackpropSelu

num_features = 10
size = 1
num_imag_in_row = 1
num_cols = num_features // num_imag_in_row
display_grid = np.ones((size * num_cols, num_imag_in_row * size))

k = 1
j = 1
for col in range(num_cols): 
  for row in range(num_imag_in_row):
    channel_image = CONV5_layer_activation[j, k, col * num_imag_in_row + row]
    display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
  
scale = 1. / size
plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
plt.title(vis_true_labels[index])
plt.grid(False)
plt.imshow(display_grid, aspect='auto',cmap='gray')  
plt.show()                                         


# Guided backprop gradient visualisation

grad_image = np.dstack((
            target_gradient[:, :, 0],
            target_gradient[:, :, 1],
            target_gradient[:, :, 2],
        ))       
grad_image = grad_image - np.min(grad_image)
grad_image = grad_image/grad_image.max()
imgplot = plt.imshow(grad_image)
plt.axis("off")
plt.show()
```

![gradcam](https://user-images.githubusercontent.com/48018142/163724025-e33dd226-8f2b-4334-a2df-c11cd41bc3bd.png)
![arch](https://user-images.githubusercontent.com/48018142/163723815-2a432c07-bc55-4b27-bdb8-3e6b873351a1.png)

