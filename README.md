# GaitML_SCI_CES_Classification

This repository contains sample gait motion data and basic documentation for a study on:

> "Development of Machine Learning Models for Gait-based Classification of Incomplete Spinal Cord Injuries (SCI) and Cauda Equina Syndrome (CES)"

## Files

- `sample_data.csv`: Sample gait data containing spatiotemporal parameters from left, right, and normalized gait cycles.
- `README.md`: This file.

## Description

Each row in `sample_data.csv` represents gait metrics from one subject, with features extracted from both limbs. The dataset includes:

- Demographics: `Sex`, `Age`
- Spatiotemporal features: cadence, step length, stride time, support phase durations, etc.
- Normal gait reference values
- `target` column indicating class label (e.g., 0: healthy, 1: CES, 2: SCI)

## Purpose

This sample is intended to support reproducibility of the gait-based machine learning classification model.

## Contact

For inquiries, contact [sg010421@gmail.com]
