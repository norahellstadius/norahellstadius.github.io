# Plant Classifier to identify out of distribution images

<div align="center"><div style="background-color: lightgrey; padding: 5px;"><strong>Big Picture Idea</strong></div></div>
<div align="center"><div style="background-color: lightgrey; padding: 5px;">Alert farmer that they should retake the image.</div></div>
<br>

<div align="center"><div style="background-color: lightgrey; padding: 5px;"><strong>Goal of This Project</strong></div></div>
<div align="center"><div style="background-color: lightgrey; padding: 5px;">Train a classifier to identify out-of-distribution images.</div></div>
<br>

<div align="center"><div style="background-color: lightgrey; padding: 5px;"><strong>Motivation</strong></div></div>
<div align="center"><div style="background-color: lightgrey; padding: 5px;">Improve the performance of all deployed models.</div></div>
<br>

## Overview of the pipline

![Pipeline](./pipline.png)

### Experimented with different architectures to find best

Three different architectures were explored, all utilizing the ResNet50 model as the backbone .Each model introduces variations in terms of the number of dense layers, the inclusion of batch normalization and dropout, and the utilization of convolutional layers with max pooling. 


1. Model 1 (V1):
- Architecture: This model consists of a single dense layer with 150 nodes along with a final classification layer.
- Techniques for avoiding overfitting: Batch normalization and dropout are applied to prevent overfitting, ensuring better generalization.

2. Model 2 (V2):
- Architecture: This model is more complex, comprising a total of four dense layers, including the final classification layers, built on the ResNet50 backbone.
- Techniques for avoiding overfitting: Similar to Model 1, batch normalization is employed to improve the generalisation.

3. Model 3 (V3):
- Architecture: In this model, a combination of convolutional layers and max pooling is added along with a dense layer on top of the ResNet50 backbone.
- Convolutional layers and max pooling: These layers allow the model to extract spatial features from the input data effectively.

|                         |                          |                          |
|-------------------------|--------------------------|--------------------------|
| <div style="background-color: #f1f1f1; padding: 10px;"> Model 1 (V1):
- Architecture: This model consists of a single dense layer with 150 nodes along with a final classification layer.
- Techniques for avoiding overfitting: Batch normalization and dropout are applied to prevent overfitting, ensuring better generalization.</div> | <div style="background-color: #f1f1f1; padding: 10px;"> Model 2 (V2):
- Architecture: This model is more complex, comprising a total of four dense layers, including the final classification layers, built on the ResNet50 backbone.
- Techniques for avoiding overfitting: Similar to Model 1, batch normalization is employed to improve the generalisation.</div> | <div style="background-color: #f1f1f1; padding: 10px;"> Model 3 (V3):
- Architecture: In this model, a combination of convolutional layers and max pooling is added along with a dense layer on top of the ResNet50 backbone.
- Convolutional layers and max pooling: These layers allow the model to extract spatial features from the input data effectively.</div> |


By exploring these different architectures, the aim is to identify the most suitable approach for the given task of image classification using ResNet50 as the backbone. 

![Summary of three models explored](./models.png)

The performance of each model was evaluated based on loss (binary cross entropy), accuracy and F1 score. Model 2 showed the best performance of all three models with a loss, accuracy and F1 score of 0.431, 0.925 and  0.953 respectively. 


| **Metrics \ Models** | **V1** | <span style="color:orange">**V2**</span> | **V3** |
|:--------------------:|:------:|:------:|:------:|
| **Loss**             | 0.435  | <span style="color:orange">0.431</span>  | 0.443  |
| **Accuracy**         | 0.918  | <span style="color:orange">0.925</span>  | 0.924  |
| **F1 Score**         |  0.949 | <span style="color:orange">0.953</span>  | 0.952  |

### Finetuning Best Performance Model

Procedding with model 2, the model is finetuned on images it has difficulity to classifiy as other. 

The provided production data consists of more than 400,000 images, but only a certain number of labels were provided. We filtered out the images that did not have labels and used the previously trained classifier (Model 2) to identify which images without labels were not of plants. We fed the images without labels into the classifier and retained all images with a probability of being classified as a plant between 0 and 0.6. Note that the classifier's output is the probability of an image being a plant (p(x = plant)).Next, we manually reviewed all of the retained images and extracted the images that were of something other than plants. We used those images to fine-tune the classifier.

In summary the following steps were taken: 

1. Extract images from the production dataset that do not have labels.
2. Select a random sample of 5000 images without labels from the extracted images.
3. Use a pre-trained image classifier to predict the probability of each image belonging to a plant class.
4. Keep the images with a probability less than or equal to 0.6 of belonging to a plant class.
5. Visualize the selected images and manually verify which ones are non-plants.
6. Create a new dataset with the verified non-plant images and fine-tune the pre-trained model.


In the below pictures steps 2 -6 are visualised: 

### 
![Finetune classifier on images the model performance poorly on](./finetune_model.png)


Below are example of images manually labeled as other in step 5

![Images identified as other](./other.png)

In step 6, we use a weighted loss function movited by the fact that rather the farmer retake image, than feed low-quality uncertain image downstream

```python
    def weighted_loss_fn(y_true, y_pred):
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight_matrix = tf.cast(tf.greater(y_true, y_pred), tf.float32) * 2.0 + 1.0
        weighted_bce_loss = bce_loss * weight_matrix
        return weighted_bce_loss
```

## Performance

