This repository is just to show and teach some Convolutional Neural Network and Transfer Learning methods used to solve Image Classification problems.

# Butterfly classification (CNN - Transfer Learning)

<p align="center">
<img src="Notebooks/images/butterflies.PNG" width=800 class="center">
</p>
  
In this repository, and based in a huge butterfly [image classified databased](https://www.research-collection.ethz.ch/handle/20.500.11850/365379), I train a model to do this classification in a autonomous way with the best possible accuracy. To achieve this, I use the help of the Pre-Trained Model-ResNet-50, it's basically a convolutional neural network, 50 layers deep, that has been trained on more than a million images from the ImageNet database. Because this is not enough o ge my specific classification, I add some layers on top of this model, to be able to trin it and get an excellent classification result.

Techniques applied:
- Data Generation
  - Data Sampling
  - Data Agumentation
- Defining class weights
- Create Model instance
- Unlocking Pre-Trained Model-ResNet-50 layers for training
- Model evaluation

<p align="center">
<img src="Notebooks/images/Structure-of-the-ResNet-50-used-for-reservoir-recognition.png" width=300 class="center">
</p>

---
### Methodology

  - **Data Understanding:** 
    - Because we actually have a clean and structured dataset, this step is just based in loading correctly all the data and geting the information we are interested in.

  - **Data Generation**
    - In this section, we will generate our data for training and testing.
    - Data Sampling, we undersample the image dataset, and also apply class weights to the trainning model because our target variable is imbalance.
 
        As you all know, imbalanced datasets can cause a problem during training because the neural network will be shown many more examples high frequent class, so it might             become better at recognizing only that class. A scikit-learn tool "Class weights" helps as to solve this problem calculating weights that will properly balance any               dataset. These weights are applied to the gradient for each image in the batch during training, so as to scale their influence on the overall gradient for the batch.

    - Data Augmentation, it's key to the training of neural networks for image classification. 

        Training CNN's for accurate predictions requires a lot of image data. Usually there is never enough images to cope with all of the variability that the classifier might encounter while also preventing overfitting.

        The smart solution to this is to randomly augment, or edit, the images just before they are passed to the network for training. This allows us to create a theoretically infinite amount of images and also generalise the model's behaviour.

<p align="center">
<image src="Notebooks/images/butterfly augmentation.png" width=800px/>
</p>


  - **Create Model instance:**

**Pre-Trained Model-ResNet-50**

<p align="center">
<image src="Notebooks/images/res50.PNG" width=900px/>
</p>

The Resnet model contains blocks of convolutional layers to extract features and a classfication head, or top, which contains an average pooling layer and a fully-connected (or dense) layer. If `include_top=True` then the whole model is initialized. If `include_top=False` then only the convolutional part of the model is initialized.

The keyword argument `weights='imagenet'` loads the model with weights that have been pretrainined on the imagenet dataset. This is much faster to train than initializing a model with randomly assigned weights

To use ResNet-50 model as a base for transfer learning, we need to create a new model with the final layers suited to our number of classes. There are multiple ways to do this, but we will reload the model and specifiy that we only want to load the convolutional layers. `include_top=True`

We use the Adam optimizer with a fairly low learning-rate. The learning-rate could perhaps be larger. But if you try and train more layers of the original base model (i.e. including conv layers as well), then the learning-rate should be quite low otherwise the pre-trained weights of the model could change too rapidly and it will be unable to learn.

<p align="center">
<image src="Notebooks/images/model.png" width=500px/>
</p>

  - **Fine-Tuning: Unlocking layers for training:**

In Transfer Learning the original pre-trained model is locked or frozen during training of the new classifier. This ensures that the weights of the original model will not change. One advantage of this, is that the training of the new classifier will not propagate large gradients back through the model that may either distort its weights or cause overfitting to the new dataset.

But once the new classifier has been trained we can try and gently fine-tune some of the deeper layers in the model as well. We call this Fine-Tuning.

In Keras, the trainable boolean in each layer of the original model is overrided by the trainable boolean in the "meta-layer" we call conv_layer (If it is set to non-trainable).So we will enable the trainable boolean for conv_layer and all the relevant layers in the original model. This is done by changing the trainable boolean of the layers we don't want to train to False.

  - **Train Model:**

  - **Evaluating predictions:**

After training we can also evaluate the new model's performance on the test-set using a single function call in the Keras API. However for a more comprehensive overview of our model, we use TensorFlow Profiler.

<p align="center">
<image src="Notebooks/images/evaluation.jpg" width=400px/>
</p>

  - **Validation:**

Looks like our model learned well

<p align="center">
<image src="Notebooks/images/validation.png" width=800px/>
</p>
