<section id="Introduction-section">
</section>

# Introduction

In this project our team collaborated with Babban Gona team to create a two-part solution. Firstly, we developed a classifier that serves as a preliminary stage for all computer vision models. Secondly, we tackled the challenge of soil health assessment by implementing semantic segmentation and object detection techniques. 

<br>
<div style="text-align:center; border: 2px solid #4285f4; padding: 10px;">
  <h2>Out of Distribution Classifier</h2>
  <p>In the exploration of production data, it was discovered that the presence of non-plant images was adversely affecting data quality and reducing the accuracy of deployed models. To address this issue, a classifier was developed to filter out unsatisfactory images and ensure that only high-quality images are used for modeling. The main objective was to create a model that alerts farmers to retake images classified as out-of-distribution. The development process involved experimenting with three different architectures, all utilizing a ResNet50 backbone. The training process involved a combination of the imagenet dataset, representing the "other" class, and labeled plant images from the production data. Performance evaluation was conducted using metrics such as loss (binary cross entropy), accuracy, and F1 score. After this initial training, the models underwent further refinement. To enhance the models' performance, a fine-tuning process was employed using images the models' performed poorly on. This iterative approach aimed to specialize the model's performance to the production data and optimize its ability to accurately determine whether an image contains a plant, or not. 
</p>
</div>

<br>
<div style="text-align:center; border: 2px solid #4285f4; padding: 10px;">
  <h2>Plant Health Assesment</h2>
  <p>The objective of this part of project was to differentiate plant regions from soil and categorize plants as yellow, purple, or best practice. The Babban Gona team had previously developed an object detection model with 80% accuracy on laboratory data, but encountered difficulties when applied to production data due to the lack of labeled production data.To address this challenge, an unsupervised segmentation algorithm was utilized, employing a CNN for feature extraction and clustering, and training the CNN on loss based on the closeness of features and spatial continuity. An algorithm was then employed to mask out labels that were not plant-based based on their RGB color and green color percentage. Finally, individual plant images were input into a trained ResNet-50 for soil health classification. The proposed approach was benchmarked against the Yolov5 method used by the Babban Gona team to evaluate its performance on noisy production data. 
</p>
</div>

<br>



<section id="plant-classification-section">
</section>

# Out of Distribution Classifier 

## Overiew of the problem

<div align="center"><div style="background-color: lightgrey; padding: 5px;"><strong>Big Picture Idea</strong></div></div>
<div align="center"><div style="background-color: lightgrey; padding: 5px;">Alert farmer that they should retake the image.</div></div>
<br>

<div align="center"><div style="background-color: lightgrey; padding: 5px;"><strong>Goal of This Project</strong></div></div>
<div align="center"><div style="background-color: lightgrey; padding: 5px;">Train a classifier to identify out-of-distribution images.</div></div>
<br>

<div align="center"><div style="background-color: lightgrey; padding: 5px;"><strong>Motivation</strong></div></div>
<div align="center"><div style="background-color: lightgrey; padding: 5px;">Improve the performance of all deployed models.</div></div>
<br>

During the exploratory analysis of the production data, it was observed that the presence of non-plant images was affecting the quality of the data and subsequently reducing the accuracy of the deployed models. To address this issue, a classifier was developed and trained with the aim of filtering out unsatisfactory images and ensuring that only high-quality images are fed into the models. The primary objective is to create a system that alerts farmers to retake an image if it is classified as out-of-distribution.


## Modeling Pipeline

The development of an accurate plant-other classifier involved the following steps:


1. **Experiment with Three Different Architectures:** Three different architectures were explored, each utilizing the ResNet50 model as the backbone. These architectures introduced variations in terms of the number of dense layers, the inclusion of batch normalization and dropout, and the utilization of convolutional layers with max pooling. The objective was to identify the most suitable approach for the task of image classification using ResNet50 as the backbone.
2. **Training the Models:** The models were trained using a combination of imagenet data, which represented the "other" class, and plant images from the production data that had labels. 
3. **Evaluation of Model Performance:** The performance of each trained model was evaluated using metrics such as loss (binary cross entropy), accuracy, and F1 score.
4. **Fine-Tuning on Non-Plant Images:** The models underwent fine-tuning using a dataset comprised of images from the production data that were identified as non-plant images. 
5. **Evaluation of Model Performance:** The performance of each fine-tuned model was evaluated using the same metrics  as before. 
6. **Fine-Tuning on Dirt-Plant Images:** The models underwent fine-tuning using a dataset comprised of images of dirt for the "other" class and images of plants for the "plant" class, in order to correct for a mis-classification weakness of the models.
7. **Evaluation of Model Performance:** The final evaluation of each models' performance was analyzed using the same metrics as before. 
<div style="text-align:center;">
  <img src="images/pipline.png" alt="The pipeline used for data preprocessing and model training"/>
  <figcaption>The pipeline used for data preprocessing and model training.</figcaption>
</div>

## Experimented with different architectures to find best

<div style="border: 2px solid #4285f4; padding: 10px; text-align:center;">
  <p> The binary classifier takes the form of P(x = Plant).</p>
</div>

Three different architectures were explored, all utilizing the ResNet50 model as the backbone.Each model introduces variations in terms of the number of dense layers, the inclusion of batch normalization and dropout, and the utilization of convolutional layers with max pooling. By exploring these different architectures, the aim is to identify the most suitable approach for the given task of image classification using ResNet50 as the backbone. Below is a brief description of each model.

<div style="display: flex; justify-content: space-between;">
  <div style="background-color: lightgrey; padding: 10px; width: 30%; display: inline-block; color: black; text-align: center;">
    <h2>Model 1 (V1)</h2>
    <p><strong>Architecture:</strong> This model consists of a single dense layer with 150 nodes along with a final classification layer.</p>
    <p><strong>Techniques for avoiding overfitting:</strong> Batch normalization and dropout are applied to prevent overfitting, ensuring better generalization.</p>
  </div>
  <div style="background-color:lightgrey; padding: 10px; width: 30%; display: inline-block; color: black; text-align: center;">
    <h2>Model 2 (V2)</h2>
    <p><strong>Architecture:</strong> This model is more complex, comprising a total of four dense layers, including the final classification layers, built on the ResNet50 backbone.</p>
    <p><strong>Techniques for avoiding overfitting:</strong> Similar to Model 1, batch normalization is employed to improve generalisation.</p>
  </div>
  <div style="background-color: lightgrey; padding: 10px; width: 30%; display: inline-block; color: black; text-align: center;">
    <h2>Model 3 (V3)</h2>
    <p><strong>Architecture:</strong> In this model, a combination of convolutional layers and max pooling is added along with a dense layer on top of the ResNet50 backbone.</p>
    <p><strong>Convolutional layers and max pooling:</strong> These layers allow the model to extract spatial features from the input data effectively.</p>
  </div>
</div>

<br>

<div style="text-align:center;">
  <img src="images/models.png" alt="Summary of three models explored"/>
  <figcaption>Summary of three models explored</figcaption>
</div>

<br>

<div align="center"><div style="background-color: #ffab40; padding: 10px;display: inline-block; color: black;">
Before we proceed with model training and evaluation, we we will shed light on the rationale behind the decision of using transfer learning and specifically the use of Resnet.
</div></div>

When confronted with the task of building an effective image classifier, transfer learning emerges as an invaluable technique in the realm of deep learning. Instead of starting from scratch and training a model on an entirely new dataset, transfer learning enables us to capitalize on the knowledge and insights gained from a pre-trained model. 

Now, you may be wondering, why ResNet? ResNet, short for Residual Network, is a powerful deep learning architecture that has demonstrated state-of-the-art performance on various computer vision tasks. By leveraging the wisdom extracted from a large-scale dataset like ImageNet, we can save substantial computational resources and significantly reduce the time required for training.


## Training the Models 
<p>We employed the Imagenet dataset to represent the "other" class, while utilizing labeled production data for the "plant" class. The choice of Imagenet stemmed from time constraints, as we lacked the capacity to individually examine over 300,000 unlabeled (???) images to identify non-plant ones. The imagenet dataset was taken from <cite>Kaggle</cite> (<a href="https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000">source</a>). Consequently, we opted to employ the best-performing model to identify these images and manually review them later. By doing so, we can extract non-plant images, which are subsequently used to fine-tune the top model. This process takes place in step 5.</p>

<div style="display: flex; justify-content: space-between;">
  <div style="border: 2px solid #4285f4; padding: 10px; width: 45%; display: inline-block; text-align: center;">
    <h2 style="font-weight: bold;">Other</h2>
    <p><strong>What:</strong> Imagenet</p>
    <p><strong>Size:</strong> 1600 (train), 200 (val), 200 (test)</p>
    <img src="images/imagenet.png" alt="Other Image" width="100px" height="100px">
  </div>
  <div style="border: 2px solid #4285f4; padding: 10px; width: 45%; display: inline-block; text-align: center;">
    <h2 style="font-weight: bold;">Plant</h2>
    <p><strong>What:</strong> Labeled production data</p>
    <p><strong>Size:</strong> 1600 (train), 200 (val), 200 (test)</p>
    <img src="images/plant_img.png" alt="Plant Image" width="100" height="100">
  </div>
</div>

During training, initially first layer of the ResNet50 model is frozen. This is done to prevent the pre-trained weights from being updated during training and to focus on fine-tuning the last added layers. The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric. An early stopping callback is also defined to stop training when the validation loss stops improving after 5 epochs. In the next step, the layers of the model are made trainable, and the model is trained for 2 more epochs with a lower learning rate.For more details on the training of all three models, please refer to the <cite>train_classifier.py</cite>  python script available in the repository at (<a href="https://github.com/Harvard-IACS/Babban_Gona/blob/main/plant_other_classifier/train_classifier.py">source</a>)


## Evaluation of Model Performance

The performance of each model was evaluated based on loss (binary cross-entropy), accuracy and F1 score. Model 2 showed the best performance of all three models with a loss, accuracy and F1 score of 0.431, 0.925 and  0.953 respectively. 

<div style="margin: auto; text-align: center;">
  <div style="display: inline-block;">
    <table>
      <tr>
        <th>Metrics \ Models</th>
        <th>V1</th>
        <th><span style="color: #ffab40">V2</span></th>
        <th>V3</th>
      </tr>
      <tr>
        <td>Loss</td>
        <td>0.435</td>
        <td><span style="color: #ffab40">0.431</span></td>
        <td>0.443</td>
      </tr>
      <tr>
        <td>Accuracy</td>
        <td>0.918</td>
        <td><span style="color: #ffab40">0.925</span></td>
        <td>0.924</td>
      </tr>
      <tr>
        <td>F1 Score</td>
        <td>0.949</td>
        <td><span style="color: #ffab40">0.953</span></td>
        <td>0.952</td>
      </tr>
    </table>
  </div>
</div>

## Fine-tuning - Round One

### Room to improve: motivation for fine-tuning

While the plant-other classifier achieves a high level of accuracy, there is still room for improvement. Currently, the model incorrectly classifies dirt as a plant, which suggests that it may be focusing too much on the background of the image rather than the low-level features that distinguish plants from other objects. To address this issue, we plan to fine-tune the model using non-plant images from the production dataset. 

<div style="text-align:center;">
  <img src="images/dirt.png" alt="Images classified in incorrectly by best performed model (v2)"/>
  <figcaption>Images classified incorrectly by best performed model (v2)</figcaption>
</div>

### Fine-tuning on non-plant images from the production data

<div style="border: 2px solid  #ffab40; padding: 10px;">
    <p><strong>Note:</strong> The below was done for all models (i.e v1, v2, v3) however model 2 (i.e v2) achieved the best performance and hence we proceed by explaining this section with only focusing on the fine-tuning of model 2.</p>
</div>

The provided production data consists of more than 400,000 images, but only a certain number of labels were provided. We filtered out the images that did not have labels and used the previously trained classifier (Model 2) to identify which images without labels were not of plants. We fed the images without labels into the classifier and retained all images with a probability of being classified as a plant between 0 and 0.6. Recall that the classifier's output is the probability of an image being a plant (i.e P(x = plant)). Next, we manually reviewed all of the retained images and extracted the images that were of something other than plants. We used those images to fine-tune the classifier.

In summary the following steps were taken: 

1. **Get Images with no labels:** Extract images from the production dataset that do not have labels.
2. **Select random sample:** Select a random sample of 5000 images without labels from the extracted images.
3. **Use trained classfier:** Use a pre-trained image classifier to predict the probability of each image belonging to a plant class.
4. **Select images classified as non plant:** Keep the images with a probability less than or equal to 0.6 of belonging to a plant class.
5. **Manually label:** Visualize the selected images and manually verify which ones are non-plants.
6. **Fine-tune model:** Create a new dataset with the verified non-plant images and fine-tune the pre-trained model.


In the jupyter notebook <cite>Find_OtherImgs.ipynb</cite> (<a href="https://github.com/Harvard-IACS/Babban_Gona/blob/main/train_production_classifier/Find_OtherImgs.ipynb">source</a>) you can follow steps 1-6 in more detail.


<div style="text-align:center;">
  <img src="images/finetune_model.png" alt="Image visualizing the above 2-6 steps"/>
  <figcaption>Image visualizing the above 2-6 steps</figcaption>
</div>

### Evaluation of Model Performance
<div style="margin: auto; text-align: center;">
  <div style="display: inline-block;">
    <table>
      <tr>
        <th>Metrics \ Models</th>
        <th>V1 fine-tuned</th>
        <th><span style="color: #ffab40">V2 fine-tuned</span></th>
        <th>V3 fine-tuned</th>
      </tr>
      <tr>
        <td>Loss</td>
        <td>0.435</td>
        <td><span style="color: #ffab40">0.431</span></td>
        <td>0.443</td>
      </tr>
      <tr>
        <td>Accuracy</td>
        <td>0.918</td>
        <td><span style="color: #ffab40">0.925</span></td>
        <td>0.924</td>
      </tr>
      <tr>
        <td>F1 Score</td>
        <td>0.949</td>
        <td><span style="color: #ffab40">0.953</span></td>
        <td>0.952</td>
      </tr>
    </table>
  </div>
</div>

### Implementation Details
#### Guidance for manual labelling

During the manual labeling process (Step 5), the following guidelines were used to classify an image as "Other":
1. Any image that does not contain a plant.
2. Any image that would not be suitable for the downstream task.

While the first point is straightforward, the second point is more subjective and requires some judgment. To implement the second point, we excluded images that were either too blurry or too zoomed-in, as they were not suitable for further analysis.

<div style="display: flex; justify-content: space-between;">
  <div style="border: 2px solid #4285f4; padding: 10px; width: 45%; display: inline-block; text-align: center;">
    <h2 style="font-weight: bold;">No Plant Image</h2>
    <p>Image not containing a plant</p>
    <img src="images/noPlant.png" alt="no plant" width="100px" height="100px">
  </div>
  <div style="border: 2px solid #4285f4; padding: 10px; width: 45%; display: inline-block; text-align: center;">
    <h2 style="font-weight: bold;">Bad Quality Plant Image </h2>
    <p>Image that contains a plant but not of good quality for the downstream task </p>
    <img src="images/badQuality.png" alt="bad plant" width="100" height="100">
  </div>
</div>
<br>
<br>
<div style="text-align:center;">
  <img src="images/other.png" alt="Images that were manually labeled as 'other' in step 5 and clearly satisfy guideline number 1"/>
  <figcaption>Sample of images that were manually labeled as 'other' in step 5 and clearly satisfy guideline number 1</figcaption>
</div>

#### Fine-tuning training details

In step 6, we use a weighted loss function motivated by the fact that rather the farmer retake image, than feed low-quality uncertain image downstream

```python
    def weighted_loss_fn(y_true, y_pred):
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight_matrix = tf.cast(tf.greater(y_true, y_pred), tf.float32) * 2.0 + 1.0
        weighted_bce_loss = bce_loss * weight_matrix
        return weighted_bce_loss
```

More details of how the classifiers were fine-tuned can be found in the python script <cite>finetune_classifiers.py</cite> (<a href="https://github.com/Harvard-IACS/Babban_Gona/blob/main/plant_other_classifier/finetune_classifiers.py">source</a>)

## Fine-tuning - Round Two 

### Room to improve (round 2): motivation for fine-tuning (again)
As previously mentioned, the model confuses dirt with plants. Although the first-round of fine-tuning, improved the models' performance. We identified a way of repurposing the other model that was developed in this project, to improve the models even further. We used the segmentation model, that was created for soil health, to identify images that contained a large region of dirt in the images. We then cropped these images to remove plants. These dirt images were then used as the "other" class in the second iteration of fine-tuning.

### Fine-tuning on dirt images identified by the segmentation model
In order to complete the second round of fine-tuning, the following steps were taken:
1. **Get images with large dirt region:** Using the segmentation model, to extract images from the production dataset that contain a large region of dirt.
2. **Crop images:** Crop the images to remove plants, creating a dataset of dirt images.
3. **Fine-tune model:** Fine-tune the model from the last round of fine-tuning using the new dataset, in an attempt to correct the model from classifying dirt as plant.

The code for this can be found in the file <a href="https://github.com/Harvard-IACS/Babban_Gona/blob/main/plant_other_classifier/finetune_dirt.py">finetune_dirt.py</a>.
### Evaluation of Model Performance
<div style="margin: auto; text-align: center;">
  <div style="display: inline-block;">
    <table>
      <tr>
        <th>Metrics \ Models</th>
        <th>V1 fine-tuned (R2)</th>
        <th><span style="color: #ffab40">V2 fine-tuned (R2)</span></th>
        <th>V3 fine-tuned (R2)</th>
      </tr>
      <tr>
        <td>Loss</td>
        <td>0.435</td>
        <td><span style="color: #ffab40">0.431</span></td>
        <td>0.443</td>
      </tr>
      <tr>
        <td>Accuracy</td>
        <td>0.918</td>
        <td><span style="color: #ffab40">0.925</span></td>
        <td>0.924</td>
      </tr>
      <tr>
        <td>F1 Score</td>
        <td>0.949</td>
        <td><span style="color: #ffab40">0.953</span></td>
        <td>0.952</td>
      </tr>
    </table>
  </div>
</div>

## How to use the final model
The trained models can be downloaded from Google Drive using the following <a href="https://drive.google.com/drive/folders/1DTMsCTwV_C5tZvaf3JcWsFjEi17vn51M">link</a>. The models from the 3 rounds of training are found in the folders <cite>first_iteration_training</cite>, <cite>second_iteration_training</cite>, and <cite>final</cite>. 
The model from the first round of training can be loaded using the following code,
```python 
path_model = ".../.../modelPath.h5"
model = tf.keras.models.load_model(path_model) 
```
Whilst, the models from the second and third round of training can be loaded using the following code,
```python
path_model = ".../.../modelPath.h5"
model = tf.keras.models.load_model(path_model, custom_objects={'weighted_loss_fn': weighted_loss_fn, 'f1': f1_fn,
                                                                 'accuracy': accuracy})
```

## Interpretability

Understanding and trusting machine learning models can often be a challenging task. However, there are techniques available to enhance interpretability, such as the utilization of saliency maps generated using GradCAM heatmaps. Before we analyse our  saliency maps, let's gain a better understanding of what GradCAM is and how it works.

<div align="center">
  <div style="background-color: #ffab40; padding: 10px; display: inline-block; color: black;">
    <h2>What is GradCAM?</h2>
    <p>GradCAM, short for Gradient-weighted Class Activation Mapping, is a visualization technique employed to interpret and comprehend the decision-making process of a convolutional neural network (CNN). By employing GradCAM, we can identify and highlight the significant regions or features within an input image that contribute most significantly to the network's prediction. The primary objective of utilizing GradCAM heatmaps is to provide visual explanations for the model's predictions by emphasizing the regions of the input image that exert the greatest influence on a particular prediction. This methodology aids in comprehending the model's decision-making process and identifying which portions of the input image are most relevant for a specific prediction.</p>
  </div>
</div>

Below we see the saliency maps of images classified as "other" and images classified as "plants." The saliency maps generated for these images reveal interesting insights into the model's behavior.Upon observing the saliency map for the image classified as a plant, we can ascertain that the model accurately focuses on the relevant part, namely the plant, in order to classify it correctly. This suggests that the model recognizes the distinctive features of plants and utilizes them to make accurate predictions.
Similarly, when examining the saliency map for the image classified as "other," we notice that the model concentrates on the head of the motor cycle.

<div style="text-align:center;">
  <img src="images/saliency_plant.png" alt="Image visualizing the above 2-6 steps"/>
  <figcaption>Saliency maps for images classified as plant</figcaption>
</div>


<div style="text-align:center;">
  <img src="images/saliency_other.png" alt="Image visualizing the above 2-6 steps"/>
  <figcaption>Saliency maps for images classified as other</figcaption>
</div>


<section id="plant-health-section">
</section>

# Plant Health Assesment

## Overiew of the problem