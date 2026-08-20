---
layout: page
title: Multi-Task Multi-Agent RL using Shared Distilled Policies
description: Adapting DeepMind's Distral to a multi-agent setting, as part of the Multi-Agent AI (COMP0124) module at UCL.
img: assets/img/projects/multidistral_featured.png
importance: 4
category: coursework
related_publications: false
---

Multi-task learning has been extensively applied to reinforcement learning problems to counteract its data inefficiency. One possible approach is based on transferring knowledge via policies: due to the similarity between tasks, an agent's policy for one task is likely to share similarities with its policy for other tasks. This idea has been applied to single-agent RL to create the [Distral](https://arxiv.org/abs/1707.04175) framework, in which an agent aims to learn a policy for each task while being constrained to find a policy similar to a shared policy.

Based on this idea, we propose MultiDistral, an extension of Distral to the multi-agent setting. We show that MultiDistral outperforms a Q-Learning baseline given few games played. However, we observe that learning with MultiDistral in a semi-collaborative setting results in one player's performance worsening as more iterations are run. We hypothesise from behaviour simulations that this is due to competitiveness, as the better player learns to dominate the competitive resource, leaving the disadvantaged player unable to access it and thus learn. We finish by studying the differences between two versions of this framework, and by analysing the impact of the different tasks' characteristics on the learning process.

[Code](https://github.com/maxjappert/multi-agent_distral) &middot; [Report]({{ site.baseurl }}/assets/pdf/multidistral_report.pdf)
