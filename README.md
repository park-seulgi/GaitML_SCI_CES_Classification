# GaitML_SCI_CES_Classification

This repository contains sample gait motion data and basic documentation for a study on:

> "Development of Machine Learning Models for Gait-based Classification of Incomplete Spinal Cord Injuries (SCI) and Cauda Equina Syndrome (CES)"

## Files

- `sample_data.csv`: Sample gait data containing spatiotemporal parameters from left, right, and normalized gait cycles.
- `README.md`: This file.

## Description

Each row in `sample_data.csv` represents gait metrics from one subject, with features extracted from both limbs. The dataset includes:

- Demographics: `Sex`, `Age`
- Spatiotemporal features: Lt_Cadence	Lt_Double_Support	Lt_Foot_Off	Lt_Limp_Index	Lt_Opposite_Foot_Contact	Lt_Opposite_Foot_Off	Lt_Single_Support	Lt_Step_Length	Lt_Step_Time	Lt_Step_Width	Lt_Stride_Length	Lt_Stride_Time	Lt_Walking_Speed	Rt_Cadence	Rt_Double_Support	Rt_Foot_Off	Rt_Limp_Index	Rt_Opposite_Foot_Contact	Rt_Opposite_Foot_Off	Rt_Single_Support	Rt_Step_Length	Rt_Step_Time	Rt_Step_Width	Rt_Stride_Length	Rt_Stride_Time	Rt_Walking_Speed	Nm_Cadence	Nm_Double_Support	Nm_Foot_Off	Nm_Limp_Index	Nm_Opposite_Foot_Contact	Nm_Opposite_Foot_Off	Nm_Single_Support	Nm_Step_Length	Nm_Step_Time	Nm_Step_Width	Nm_Stride_Length	Nm_Stride_Time	Nm_Walking_Speed
- `target` column indicating class label (e.g., 0: Incomplete tetraplegia, 1: Incomplete paraplegia, 2: Cauda equina syndrome )

## Purpose

This sample is intended to support reproducibility of the gait-based machine learning classification model.

## Contact

For inquiries, contact [sg010421@gmail.com]
