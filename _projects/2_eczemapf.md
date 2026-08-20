---
layout: page
title: "EczemaPF: An Online Learning Approach to Real-Time Eczema Severity Prediction"
description: Undergraduate Thesis at the Tanaka Group (Imperial College London).
img: assets/img/projects/eczemapf_featured.png
importance: 2
category: research
related_publications: false
---

Note: this work is currently being written as a paper with several other authors, so many details have been omitted from this page until publication.

Eczema (atopic dermatitis, or AD) is a chronic skin disease, for which designing personalised treatment strategies, as opposed to "one-size-fits-all" approaches, is of high clinical relevance. Designing personalised treatment strategies requires an accurate prediction of the evolution of eczema severity because disease symptoms fluctuate dynamically every day and manifest as relapses and remissions. Past work (EczemaPred) has used Markov-Chain Monte Carlo, an offline learning method, to predict the evolution of eczema severity scores. However, due to the offline nature of the learning algorithm, the time required for prediction grows with the number of days in the time-series data. For a time-series of about 42 days, past work takes approximately 40 seconds per patient to output a daily prediction — a number that grows without bound as the time-series and dataset grow, making it unsuitable for near real-time prediction that can guide patients towards daily treatment.

To address this problem, I propose EczemaPredFast (EczemaPF), an online inference framework that can output predictions in a matter of seconds. This framework uses Sequential Monte Carlo to sequentially fit the model to the data as it becomes available, and to predict a patient's AD severity scores in the upcoming days. EczemaPF outperformed all reference models for 4-days-ahead PO-SCORAD prediction in both datasets after initial training, while predicting a patient's AD severity for the next four days in less than a second on average. In contrast, EczemaPred's runtime grew without bound as the dataset size increased — reaching over 10 hours for a dataset of PO-SCORAD scores over 77 days for 336 patients. In the future, this framework could be implemented as a severity management tool, providing patients with a prognosis of their eczema, as well as treatment recommendations.
