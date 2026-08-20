---
layout: page
title: Impact of the Pre-Training Data Distribution on the Fine-Tuned Performance of Masked Autoencoders
description: Group project as part of the Applied Deep Learning (COMP0197) module at UCL.
img: assets/img/projects/adl_group_featured.png
importance: 3
category: coursework
related_publications: false
---

Self-supervised learning approaches have gained significant popularity in recent years, specifically when used for the pre-training of very large foundation models, both in NLP and Computer Vision. These methods allow for pretraining on very large amounts of data (for which collection of labels would be impractical), resulting in models capable of learning richer embedded representations. These models can then be fine-tuned to specific tasks using significantly less data than if trained from scratch.

In computer vision, these methods are mostly divided into invariance-based methods and generative methods. The former are based on training an encoder-like network to produce similar embeddings for images of the same scene but with different views — the idea behind contrastive learning. Generative methods (which include masked autoencoders, or MAEs) are based on corrupting portions of the input images and learning to predict these corrupted portions; in doing so, the model learns meaningful representations, albeit of a lower level than contrastive methods.

A natural question that arises from pre-training large models is how the distribution of the pre-training data affects the model's performance on downstream tasks. In this project, we hypothesise that pre-training on data from a distribution similar to the fine-tuning data should result in better model performance. We implement a Masked Autoencoder and pre-train it on the MS COCO dataset, then fine-tune it to perform image segmentation on the Oxford Pet dataset. Different splits of the MS COCO pre-training dataset are used to study the impact of the pre-training data distribution on the downstream segmentation results.

[Code](https://github.com/Sam-Oliveira/pretraining_mae) &middot; [Report]({{ site.baseurl }}/assets/pdf/adl_report.pdf)
