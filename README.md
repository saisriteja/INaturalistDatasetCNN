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
![truelabel](https://user-images.githubusercontent.com/48018142/163723275-8a14470a-1fc7-44f6-9ad2-8d7f33d5d62b.JPG)
