# Project-2-Deep-Learning
## Week-5-Transfer Learning with ResNet-18

Project Overview

This project explores convolutional neural networks (CNNs), image processing, and transfer learning using PyTorch. A pretrained ResNet-18 model is adapted to classify natural scene images from the Intel Image Classification dataset.

The project includes image visualization, pixel analysis, convolution operations, parameter calculations, data augmentation, feature extraction, fine-tuning, model evaluation, and error analysis.

Objectives

* Understand image representation and RGB pixel values.
* Explore convolutional filters and edge detection.
* Calculate parameters in convolutional and dense layers.
* Prepare image datasets using PyTorch and data augmentation.
* Apply transfer learning with a pretrained ResNet-18 model.
* Improve classification performance through fine-tuning.
* Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
* Save the trained model for future use.

Dataset

Dataset: Intel Image Classification

The dataset contains six scene categories:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

Dataset splits used in this project:

Split	Number of images
Training	11,929
Validation	2,105
Testing	3,000

The original training dataset contains 14,034 images. A 15% validation split was created from the training data.

Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow (PIL)
* Scikit-learn
* Kaggle Notebooks

Project Workflow

1. Image Visualization and Pixel Analysis

Displayed sample images from all six classes and examined image dimensions, RGB channels, and pixel values.

2. Convolution and Parameter Analysis

Demonstrated vertical edge detection using a convolution kernel and calculated the parameters of convolutional and fully connected layers.

3. Data Preparation and Augmentation

Applied random resized cropping, horizontal flipping, and color jittering to training images. Validation and test images were resized and normalized using ImageNet statistics.

4. Transfer Learning — Feature Extraction

Loaded pretrained ResNet-18 weights and replaced the final fully connected layer with a six-class classification head. The pretrained feature-extraction layers were frozen during the initial training stage.

* Optimizer: Adam
* Learning rate: 0.001
* Epochs: 3
* Best validation accuracy: 90.1%

5. Fine-Tuning

Unfroze the final ResNet-18 convolutional block (layer4) and trained it with the new classification head using separate learning rates.

* Learning rate for layer4: 0.0001
* Learning rate for classification head: 0.001
* Fine-tuning epochs: 3
* Best validation accuracy: 93.2%

6. Model Evaluation and Error Analysis

Evaluated the fine-tuned model on 3,000 test images. Generated a classification report, confusion matrix, and examples of incorrect predictions.

7. Model Saving

Saved the trained model’s state dictionary as resnet18_intel.pt.

Results

Metric	Result
Test Accuracy	93.5%
Macro Precision	93.7%
Macro Recall	93.5%
Macro F1-score	93.6%
Weighted F1-score	93.4%
Best Validation Accuracy after Fine-Tuning	93.2%

The forest class achieved the strongest F1-score at 0.989. Some errors occurred between visually similar classes, particularly glaciers and mountains, and buildings and streets.

Model File

The trained model was saved at:

/kaggle/working/resnet18_intel.pt

The saved file is approximately 44.8 MB. To make it available to other users, download it from Kaggle’s output files and upload it to the GitHub repository or a suitable model-hosting service.

How to Run

1. Open Kaggle and create a notebook.
2. Add the Intel Image Classification dataset.
3. Enable GPU acceleration if available.
4. Run the notebook cells in order, from imports and dataset preparation through training and evaluation.
5. Review the accuracy metrics, learning curves, confusion matrix, and misclassified examples.
6. Download the saved model from the notebook’s output files if you want to reuse it.

The notebook uses the dataset path configured for Kaggle. If running elsewhere, update the dataset paths accordingly.

Conclusion

This lab demonstrates that transfer learning can achieve strong image classification performance without training a deep neural network entirely from scratch. Fine-tuning the final convolutional block improved the model’s validation accuracy from approximately 90.1% to 93.2%. The final model achieved 93.5% test accuracy on the Intel Image Classification dataset.

Author

Name: Add your name here

Course/Lab: Week 5 — Transfer Learning, Lab 5

Acknowledgements

* Intel Image Classification dataset
* PyTorch and Torchvision
* Scikit-learn documentation

Week 6 Proposal: Satellite Image Classification (Next Week)

About This Project

This project uses transfer learning with a pretrained ResNet-18 model to classify satellite images into four categories: cloudy, desert, green area, and water. It builds on the Week 5 lab and tests how well features learned from everyday photos work on top-down satellite views.

Proposal Overview

Next week's project applies the transfer learning workflow from Week 5 to satellite imagery. A pretrained ResNet-18 model will be adapted to classify satellite images into four terrain and environment categories using a Kaggle image dataset. The project will test how well ImageNet features transfer to top-down satellite views.

Problem Statement

Can a transfer-learned CNN accurately classify satellite images as cloudy, desert, green area, or water, and which classes does it confuse most?

Proposed Dataset

Dataset: Satellite Image Classification (Kaggle)

Link: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

This dataset contains 5,631 satellite images in four classes: Cloudy (1,500), Desert (1,131), Green Area (1,500), and Water (1,500). Images are organized in class folders. Since no official train/validation/test split is provided, a stratified split of about 70% training, 15% validation, and 15% testing will be created. Images will be resized to 224x224 and normalized using ImageNet statistics.

Objectives

* Load and explore the satellite images and check class balance.
* Create a stratified train/validation/test split.
* Apply augmentation suited to satellite images (horizontal and vertical flips, 90-degree rotations, mild color jitter).
* Train a pretrained ResNet-18 with a new four-class head (feature extraction), then fine-tune layer4, as in Week 5.
* Compare ResNet-18 with a second model such as ResNet-50 or EfficientNet-B0.
* Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix.
* Analyze misclassified images, especially between visually similar classes.

Planned Methodology

1. Data preparation: stratified split, resizing, augmentation, and ImageNet normalization.
2. Feature extraction: freeze the pretrained layers and train the new head (Adam, learning rate 0.001).
3. Fine-tuning: unfreeze layer4 (learning rate 0.0001) and train it with the head (learning rate 0.001).
4. Model comparison: repeat the best setup with a second architecture.
5. Evaluation and error analysis on the held-out test set.
6. Save the final model as resnet18_satellite.pt.

Technologies

* Python
* PyTorch and Torchvision
* NumPy
* Matplotlib
* Pillow (PIL)
* Scikit-learn
* Kaggle Notebooks (GPU)

Evaluation Metrics

Metric	Purpose
Accuracy	Overall performance
Macro Precision, Recall, F1-score	Performance with all classes weighted equally
Per-class F1-score	Shows which terrain types are hardest
Confusion matrix	Shows which classes are mixed up

Expected Outcomes

* High test accuracy, with fine-tuning improving on feature extraction alone (target to be confirmed after training).
* A clear comparison between the two models.
* Error analysis, with attention to green area and water images, which can look similar from above.

Risks and Limitations

* The dataset is small and has only four broad classes.
* Some images may come from different sources (sensor data mixed with map snapshots), which can affect consistency.
* ImageNet pretraining uses ground-level photos, which differ from satellite views.

Proposed Timeline

Stage	Task
Day 1	Download dataset, explore data, create split
Day 2	Build data pipeline and train feature-extraction baseline
Day 3	Fine-tune layer4
Day 4	Train second model and compare
Day 5	Evaluation, confusion matrix, and error analysis
Day 6	Write results and save the model

Author

Name: Hussain Mohiuddin

Course/Lab: Week 6 — Project Proposal

Proposal Acknowledgements

* Satellite Image Classification dataset (Kaggle)
  ## Project 2:Week-6 Satellite Image Classification Using Deep Learning

Author: Hussain Mohiuddin
Course/Lab: Applied AI, Week 6
Project Type: Multiclass Image Classification

1. Project Overview

This project focuses on classifying satellite images into four categories using deep learning and transfer learning. A pretrained ResNet-18 model will be used as the baseline, followed by fine-tuning and comparison with a second architecture.

The goal is to investigate how well deep learning models distinguish different land and environmental conditions in satellite imagery.

2. Dataset

Dataset Name: Satellite Image Classification

Source: Kaggle

Dataset Link: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

Total Images: 5,631

Classes and Distribution

Class	Number of Images
Cloudy	1,500
Desert	1,131
Green Area	1,500
Water	1,500
Total	5,631

The dataset contains four categories organized into class folders. Desert has fewer images than the other classes, so class distribution will be checked before training.

3. Objectives

* Explore satellite images and examine class distribution.
* Create stratified training, validation, and testing splits.
* Resize images to 224 × 224 pixels and normalize them using ImageNet statistics.
* Apply suitable data augmentation, including horizontal and vertical flips, 90-degree rotations, and mild color jitter.
* Train a pretrained ResNet-18 model using feature extraction.
* Fine-tune the final ResNet block to improve performance.
* Compare ResNet-18 with a second architecture, such as ResNet-50 or EfficientNet-B0.
* Evaluate models using accuracy, precision, recall, Macro-F1, and a confusion matrix.
* Inspect misclassified images to understand common errors.
* Save the selected model for future inference.

4. Planned Methodology

Step 1: Data Preparation

Load the images, inspect class counts, resize them to 224 × 224 pixels, and create a stratified split.

The planned split is approximately 70% training, 15% validation, and 15% testing.

For 5,631 images, the approximate counts are:

* Training: 3,942 images
* Validation: 845 images
* Testing: 844 images

Exact counts may vary slightly depending on the splitting method.

Step 2: Feature Extraction

Load pretrained ResNet-18 weights and replace the final classification layer with a new four-class output layer.

Freeze the pretrained feature-extraction layers and train the new classification head using Adam with a learning rate of 0.001.

Step 3: Fine-Tuning

Unfreeze the final ResNet block and train it with the classification head.

The planned learning rates are:

* Final ResNet block: 0.0001
* Classification head: 0.001

Compare validation performance with the feature-extraction baseline.

Step 4: Model Comparison

Train and evaluate a second architecture using the same dataset splits and evaluation procedure.

Compare the models using validation Macro-F1 and other relevant metrics. Select the best model before evaluating the held-out test set.

Step 5: Evaluation and Error Analysis

Report test accuracy, precision, recall, Macro-F1, per-class performance, and the confusion matrix.

Inspect incorrect predictions, particularly cases where green areas and water appear visually similar.

Step 6: Save the Model

Save the selected model weights and the class-label mapping.

Planned output filename: resnet18_satellite.pt

5. Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow (PIL)
* Scikit-learn
* Kaggle Notebooks with GPU acceleration

6. Evaluation Metrics

Metric	Purpose
Accuracy	Measures the proportion of correct predictions
Precision	Measures how many predictions for a class are correct
Recall	Measures how many actual examples of a class are identified
Macro-F1	Gives equal importance to each class
Confusion Matrix	Shows which classes are confused with one another

Macro-F1 is particularly useful because the number of Desert images is lower than the counts for the other classes.

7. Expected Outcomes

* A trained satellite image classification model.
* A comparison between feature extraction and fine-tuning.
* A performance comparison between two CNN architectures.
* A confusion matrix and analysis of common classification errors.
* A saved model suitable for experimentation and future inference.

These are expected outcomes, not measured results. Final performance will be reported after training and testing.

8. Risks and Limitations

* The dataset contains a limited number of images compared with large-scale remote-sensing datasets.
* The class distribution is not perfectly balanced.
* Images from different sources may have different visual characteristics.
* ImageNet pretraining may not capture every feature specific to satellite imagery.
* Performance on unseen geographic regions and different satellite sensors has not yet been established.

9. Proposed Timeline

Day	Task
Day 1	Download dataset, explore images, and create splits
Day 2	Build the data pipeline and train the feature-extraction baseline
Day 3	Fine-tune the final ResNet block
Day 4	Train a second model and compare architectures
Day 5	Evaluate the selected model and perform error analysis
Day 6	Document findings and save the final model

10. Conclusion

This project will explore the use of transfer learning for satellite image classification across four environmental categories. By comparing feature extraction, fine-tuning, and a second model architecture, the project aims to identify a reliable baseline and understand its strengths and limitations.

The final conclusions will be based on measured test results, class-wise performance, and an analysis of misclassified images.

References

* Dataset: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
* PyTorch: https://pytorch.org/
* Torchvision: https://pytorch.org/vision/stable/index.html
