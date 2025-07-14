# Flower Classification Project

 Here’s what this repo is all about, why I’m doing it, and what I’ve learned along the way.

---

## Purpose

I wanted to get hands-on with modern machine learning workflows, not just building a model, but also tracking experiments, handling data, and making the whole thing reproducible. Flowers are a classic dataset—colorful, varied, and just complicated enough to be interesting.

---

## Goal

The main goal is to build a robust image classifier and try different approaches to see how it would influence the result. I wanted to:

- Try out different model architectures (ResNet, ViT, Inception, and a simple CNN)
- Use data augmentation and regularization tricks (like CutMix and MixUp)
- Track experiments and results with MLflow
- Make the code modular and reusable
- Learn how to properly log, load, and serve models

---

## Approach

- **Data:** Used the classic flowers dataset, loaded with PyTorch’s `ImageFolder`.
- **Transforms:** Spent a lot of time tweaking data augmentations—random crops, flips, normalization, and even some Gaussian noise. Got clearly threshold of overfitting and underfitting the model with the usage of different transforms techniques
- **Models:** Implemented several architectures, from simple CNNs to pretrained ResNet and ViT models.
- **Training:** Added early stopping, learning rate scheduling, and experimented with different optimizers.
- **Experiment Tracking:** Used MLflow to log metrics, parameters, and models.
- **Evaluation:** Wrote scripts to visualize predictions, plot confusion matrices, and generally see how the models were doing beyond just accuracy.
- **Test** I photographed a couple dozen samples from the outside, most of which are not included in the basic 5 classes. Цanted to see how the models would handle the predictions.

---

## Final thoughts
Wanted just to create simple code to train model on some dataset for visualisation task(which is not my part of deep learning, actually i studied more on LLM  and NLP stuff)