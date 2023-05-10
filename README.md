<section id="Introduction-section">

<title>Introduction</title>

In this project, our team collaborated with Babban Gona to create a two-part solution. Firstly, we developed a classifier that serves as a preliminary stage for all computer vision models. Secondly, we tackled the challenge of soil health assessment by implementing semantic segmentation and object detection techniques.
<br>
<div style="text-align:center; border: 2px solid #4285f4; padding: 10px;">
  <h2>Out of Distribution Classifier</h2>
  <p>In the exploration of production data, it was discovered that the presence of non-plant images was adversely affecting data quality and reducing the accuracy of deployed models. To address this issue, a classifier was developed to filter out unsatisfactory images and ensure that only high-quality images are used for modeling. The main objective was to create a model that alerts farmers to retake images classified as out-of-distribution. The development process involved experimenting with three different architectures, all utilizing a ResNet50 backbone. The training process involved a combination of the imagenet dataset, representing the "other" class, and labeled plant images from the production data. Performance evaluation was conducted using metrics such as loss (binary cross entropy), accuracy, and F1 score. Among the trained models, the best-performing one was selected for further refinement.To further enhance the chosen model's performance, a fine-tuning process was employed, utilizing non-plant images from the production data. This iterative approach aimed to specialize the model's performance to the production data and optimize its ability to accurately classify non-plant images. 
</p>
</div>

<br>
<div style="text-align:center; border: 2px solid #4285f4; padding: 10px;">
  <h2>Plant Health Assesment</h2>
  <p>The objective of this part of project was to differentiate plant regions from soil and categorize plants as yellow, purple, or best practice. The Babban Gona team had previously developed an object detection model with 80% accuracy on laboratory data, but encountered difficulties when applied to production data due to the lack of labeled production data.To address this challenge, an unsupervised segmentation algorithm was utilized, employing a CNN for feature extraction and clustering, and training the CNN on loss based on the closeness of features and spatial continuity. An algorithm was then employed to mask out labels that were not plant-based based on their RGB color and green color percentage. Finally, individual plant images were input into a trained ResNet-50 for soil health classification. The proposed approach was benchmarked against the Yolov5 method used by the Babban Gona team to evaluate its performance on noisy production data. 
</p>
</div>

<br>
</section>





<!-- <section id="Plant-Health-section">
    <!-- content of the project 2 section -->
<!-- </section> --> 

