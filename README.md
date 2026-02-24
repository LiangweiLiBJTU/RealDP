# RealDP: Real-Time and Scalable Prediction-Driven Data Preparation Pipeline Construction

This repository contains the implementation of **RealDP**, a prediction-driven framework for automated data preparation pipeline construction.  
RealDP directly infers effective data preparation pipelines from dataset-level characteristics, enabling real-time and scalable pipeline generation without execution-driven search.

---

## Overview

Automated data preparation is critical for building high-quality machine learning pipelines, but existing approaches often rely on expensive execution-driven search.

**RealDP** reformulates pipeline construction as a prediction problem and achieves:

- Near-constant inference time
- Scalable pipeline generation
- Competitive downstream performance
- Real-time deployment capability

The framework consists of:

- Dataset-level meta-feature extraction
- Meta-learning-based task sequence inference
- Multi-stage operator prediction with ensemble models
- One-shot pipeline construction

---
