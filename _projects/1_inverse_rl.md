---
layout: page
title: Inverse RL using Generative Planning Models in Trajectory Space
description: MSc Thesis in Ilija Bogunovic's Group, as part of the MSc in Machine Learning at UCL.
img: assets/img/projects/inverse_rl_featured.png
importance: 1
category: research
related_publications: false
---

Reinforcement Learning has been applied with great success to tasks where a reward signal is clearly defined or can be hand-crafted. However, its application to tasks such as alignment to ethical standards has been limited by the inability to craft a reward function that can balance multiple (and often subjective) preferences. A possible solution is Inverse Reinforcement Learning (IRL), a class of problems in which one learns a reward function from observed agent behaviour.

In this work I propose a method for learning a reward function using diffusion models. Recent work has proposed using diffusion models to learn high-reward policies in sequential decision-making tasks. The general method involves training a diffusion model on a dataset of trajectories in order to learn a model of the environment dynamics, and then using the classifier guidance property of diffusion models to steer their output towards high-return policies. In this work I hypothesise that for a choice of trajectory similarity metric, and given a diffusion model trained on arbitrary trajectories in an environment, and example trajectories of a behaviour we wish to imitate, one can learn a proxy reward function of the desired behaviour (IRL). This learnt reward function can be used to steer the diffusion process towards the behaviour distribution, making the method learn a reward function while also imitating behaviour.

I study the performance of this method across three different environments, evaluating both the quality of the reward function learnt, as well as the quality of the output behaviour. The method learns a reward function that induces optimal behaviour in simple environments, outperforming state-of-the-art IRL methods. I extend this method to more complex environments, showing that its performance lags behind in such settings, and present reasons for the failure modes of the method along with possible fixes.

[Thesis PDF]({{ site.baseurl }}/assets/pdf/msc_thesis_inverse_rl.pdf) &middot; [Code](https://github.com/Sam-Oliveira/diffuser_irl)
