# Project-2-Deep-Learning
Transfer Learning with ResNet-18

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
