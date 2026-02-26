# Literature Review – Autonomous Driving Research

## Purpose
This folder contains my structured study of research papers related to autonomous driving systems.  
The goal is to understand state-of-the-art methods, identify system limitations, and discover research gaps for future PhD work.

---

## Papers Reviewed

### 1. A Survey of Deep Learning Techniques for Autonomous Driving
- [Paper Review File](survey_deep_learning_autonomous_driving.md)
- Focus: Technical review of deep learning architectures (modular vs. End2End) and core methodologies (CNN, RNN, DRL) for vehicle perception, localization, planning, and control.
 Evaluation of critical implementation challenges regarding functional safety, public training datasets, and computational .

### 2. End-to-End Learning for Self-Driving Cars
- [Paper Review File](EndtoEndLearning_for_self_driving_cars.md)
- Focus: Critical analysis of DAVE-2, NVIDIA's end-to-end CNN system mapping raw front-camera pixels directly to steering commands. Covers the DAVE-2 architecture (9-layer CNN, ~250K parameters), data engineering strategy (recovery augmentation, curve oversampling), closed-loop simulation evaluation, and emergent road-feature representations.
Evaluation of key limitations including temporal blindness, absence of planning, and generalization gaps — with extensions toward uncertainty estimation, multi-modal fusion, and world models.

---

## My Approach to Reading Papers
For each paper, I focus on:
- Problem addressed  
- Model architecture and methodology  
- Datasets and evaluation metrics  
- Strengths, limitations, and challenges  
- Research gaps and future directions  
- Personal research reflections

---

## Emerging Research Focus
Through reviewing multiple papers, I aim to:
- Identify where autonomous vehicles fail in real-world situations  
- Study system-level weaknesses and corner cases  
- Understand technical limitations for learning-based methods  
- Find opportunities for practical experiments and future PhD research
