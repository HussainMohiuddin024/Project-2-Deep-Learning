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
  ## Week 6: Intel Image Classification with ResNet-18

1. Project Overview

This project classifies natural scene images into six categories using transfer learning with a pretrained ResNet-18 model.

Classes: Buildings, Forest, Glacier, Mountain, Sea, Street

Purpose: To study image classification, dataset quality, data leakage risks, class imbalance, model evaluation, and visual explanations using Grad-CAM.

2. Dataset Datasheet

Dataset: Intel Image Classification

Source: Kaggle dataset by puneet6060.

Task: Multiclass image classification.

Number of classes: 6

Training and validation split: 11,695 training images and 2,339 validation images.

Test set: 3,000 images.

Image properties: RGB images resized to 150 × 150 pixels.

Class Distribution

Class	Training images before validation split
Buildings	2,191
Forest	2,271
Glacier	2,404
Mountain	2,512
Sea	2,274
Street	2,382

The dataset has a relatively balanced class distribution. The largest class contains 2,512 images, while the smallest contains 2,191 images.

Data Quality Audit

* Exact duplicate files identified: 36.
* Test images flagged as possible near-duplicates of training images: 19 out of 3,000.
* The validation split reported zero shared group IDs between training and validation.
* However, the current grouping appears to assign approximately one group per image. Therefore, this check does not establish subject-level separation.
* Possible duplicate pairs should be inspected before drawing conclusions about leakage.

3. Model Card

Model Details

Architecture: ResNet-18

Approach: Transfer learning using pretrained ImageNet weights.

Training strategy:

1. Train the classification head.
2. Fine-tune the final ResNet block and classification head.

Input size: 150 × 150 pixels.

Input channels: 3 (RGB).

Saved model: week6_model.pt

Evaluation Results

Metric	Result
Best validation Macro-F1	0.9302
Test Macro-F1	0.9152
Test accuracy	0.914
Test weighted F1	0.914

The model performs well overall, but its errors are not distributed equally across all classes.

Important Misclassifications

The confusion matrix shows that some glacier images are classified as mountains. Some street images are also classified as buildings.

These errors suggest that visually similar scenes can be difficult to distinguish.

Grad-CAM Analysis

Grad-CAM heatmaps were generated for three correct and three incorrect predictions.

In the inspected examples, the incorrect predictions were labelled as street even though their true label was buildings. Some highlighted regions included roads, ground, and surrounding structures.

Initial verdict: Shortcut learning is possible but not confirmed. More experiments are needed to establish whether the model relies on background features instead of relevant building structures.

Intended Use

This model is intended for educational experimentation and scene classification demonstrations.

It should not be treated as a safety-critical system or used as the sole basis for consequential decisions.

Limitations

* The model was evaluated on one dataset.
* Performance on images from other sources has not been established.
* Similar-looking classes can be confused.
* The duplicate audit identified potential overlap that requires further investigation.
* Grad-CAM is an interpretability aid, not proof of the model’s reasoning.
* The dataset’s licence and annotation process should be checked against the original source before redistribution.

4. Responsible AI Questions

Q1. Who could be harmed, and how?

In this scene-classification setting, incorrect predictions could affect applications such as image organization, mapping, or environmental scene analysis. A wrong class could lead to incorrect labels or downstream decisions. The model should therefore be reviewed before use in consequential applications.

Q2. Where has the model not been validated?

The model has not been validated on independent datasets, images from different regions, different cameras, or changing environmental conditions. Its generalization to these settings remains unknown.

Q3. What is the human role?

A human should review uncertain predictions, inspect important errors, and verify the model’s outputs before they are used in a real application. The model should assist rather than replace appropriate human judgment.

Q4. What evidence would change the conclusions?

Evaluation on an independent dataset, inspection of duplicate pairs, testing on varied backgrounds, and controlled experiments that mask suspected shortcut regions would provide stronger evidence about generalization and shortcut learning.

5. Conclusion

The ResNet-18 model achieved a test Macro-F1 of 0.9152 on the Intel Image Classification dataset. The dataset audit identified exact duplicates and possible near-duplicates, while Grad-CAM highlighted examples that deserve further investigation.

The main lesson is that a strong test score alone is not enough. Data quality, leakage risks, class-specific errors, and model explanations must also be examined.
