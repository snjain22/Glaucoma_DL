# Glaucoma Detection using Compact Convolutional Transformer (CCT) with Reinforcement Learning (RL)

This repository implements a novel approach for glaucoma detection using **Compact Convolutional Transformers (CCT)** integrated with **Reinforcement Learning (RL)**. The model achieves high accuracy by leveraging advanced deep learning techniques and multiple large-scale retinal image datasets.

---

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
  - [Tokenization and Positional Embeddings](#tokenization-and-positional-embeddings)
  - [Transformer Block and Regularization](#transformer-block-and-regularization)
  - [Actor-Critic Network](#actor-critic-network)
- [Training and Testing](#training-and-testing)
- [Results](#results)
- [Hardware Used](#hardware-used)
- [How to Use](#how-to-use)

---

## Introduction

Glaucoma is a leading cause of irreversible blindness worldwide. This project proposes a deep learning-based solution that combines the power of **Compact Convolutional Transformers (CCT)** with **Reinforcement Learning (RL)** to detect glaucoma from retinal fundus images with high accuracy.

Key features:
- Utilizes multiple large-scale datasets for improved generalizability.
- Implements advanced preprocessing techniques to enhance image quality.
- Achieves near state-of-the-art performance on benchmark datasets.

---

## Datasets

The model was trained on 7,761 images selected from multiple large-scale glaucoma image datasets, including:

- **BEH**: Birmingham Eye Health Study
- **CRFO**: Center for Retinal Fundus Images
- **DR HAGIS**: Annotated for both normal and glaucomatous cases
- **DRISHTI**: Optic disc and cup segmentation dataset
- **EyePACS**: Retinal images from diabetic retinopathy screening programs
- **FIVES**: Five University Eye Study dataset
- **G1020**: Annotated for glaucoma diagnosis
- **HRF**: High-resolution fundus images with detailed annotations
- **ORIGA**: Optic Nerve Head Image Annotation and Grading
- **REFUGE1**: Retinal Fundus Glaucoma Challenge Database
- **sjchoi86-HRF**: High-resolution fundus image dataset curated by sjchoi86

To ensure robustness, random combinations of these datasets were used during training. Details of the dataset combinations can be found in the below image.
We have used an incredible 50 combinations of the datasets.
![image](https://github.com/user-attachments/assets/29f57859-bc19-4b4a-a93a-8d4fa2dc1e7f)


---

## Preprocessing

Preprocessing steps include:
1. **Resizing**: Images resized to 120x120 pixels.
2. **Normalization**: Pixel values scaled to \([0, 1]\).
3. **Data Augmentation**:
   - Random rotations (\(-15^\circ\) to \(+15^\circ\)).
   - Horizontal and vertical flipping.
   - Random zooming.
   - Brightness and contrast adjustments.
4. **Image Enhancement**:
   - Histogram equalization.
   - Contrast-limited adaptive histogram equalization (CLAHE).

The dataset was split into training and validation sets using an 80/20 ratio.

---

## Model Architecture

### Tokenization and Positional Embeddings
Images are divided into patches, which are tokenized using convolutional layers followed by max pooling. Positional embeddings are added to retain spatial information.

### Transformer Block and Regularization
The core of the model is a transformer block with multi-head attention mechanisms. To prevent overfitting, stochastic depth regularization is applied:
- Probability of keeping a layer: \(0.9\)
- Probability of dropping a layer: \(0.1\)

### Actor-Critic Network
The CCT output is fed into an Actor-Critic Network:
1. **Actor Network** predicts the probability distribution over classes (normal or glaucoma).
2. **Critic Network** evaluates the action-state pair and provides feedback to refine the Actor's policy.

This integrated approach allows joint training of the CCT and RL components.

---

## Training and Testing

The model was trained using a composite loss function:
1. Policy loss (cross-entropy loss between predicted and actual actions).
2. Value loss (mean squared error between predicted and actual returns).

Optimization was performed using the Adam optimizer.

---

## Results

The model achieved high accuracy on various dataset combinations:

| Dataset Combination      | Accuracy (%) | Loss    |
|---------------------------|--------------|---------|
| BEH + FIVES + EyePACS     | 98.23        | 0.3316  |
| EyePACS + REFUGE1         | 97.90        | N/A     |


---

## Hardware Used

The experiments were conducted on the following hardware:

1. AMD Ryzen 9 5900HS CPU with NVIDIA RTX 3060 Mobile GPU (6GB).
2. Apple M2 Pro (8-core CPU + 8-core GPU).

---
