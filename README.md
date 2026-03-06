# Network-Based Risk Modeling of Alzheimer's Disease Progression

## Overview

Alzheimer’s Disease (AD) is characterized by the accumulation and propagation of misfolded proteins, including *Amyloid-β* and *tau*, across the brain in a spatially heterogeneous manner. While prior studies have identified distinct progression subtypes, the mechanisms driving heterogeneity in disease trajectories remain unclear.

This repository presents two complementary research projects that investigate Alzheimer’s disease through the lens of **network dynamical systems**, **topological data analysis (TDA)**, and **propagation modeling**. Together, these works aim to characterize disease risk, subtype heterogeneity, and underlying network constraints shaping pathological spread.

---
<p align="center">
    <img src="resources/other/brain_.png", width="700">
</p>

# Project 1: Risk Assessment in Network Dynamical Systems

## Motivation

Risk is an intrinsic part of biological aging. As the brain ages, chronic low-grade inflammation and immune system decline disrupt neural equilibrium, increasing vulnerability to neurodegenerative diseases such as Alzheimer’s Disease.

While affected brain regions and progression order are known markers of severity, a quantitative framework to evaluate **network-based risk dynamics** remains underdeveloped.

## Research Contribution

This project introduces a **novel risk framework for network dynamical systems** using tools from **Topological Data Analysis (TDA)** to study:

- First passage times to Alzheimer's pathology  
- The shape of disease propagation trajectories  
- Inflammation and related biomarkers as risk indicators  
- Structural vulnerability in brain networks  

The framework leverages:

- Persistent Homology  
- Topological summaries of dynamical trajectories  
- First passage time distributions  
- Network-based risk quantification  

### Objective

To clarify underlying disease dynamics and provide quantitative support for **preventative intervention strategies** by identifying early structural risk signals in brain networks.

---

# Project 2: Network Propagation and Heterogeneous AD Subtypes (planned/future)

## Motivation

Alzheimer’s progression follows heterogeneous spatial trajectories. Although subtypes have been identified (e.g., via SuStaIn), the mechanisms driving divergent disease paths are not fully understood.

This project investigates whether heterogeneity can be explained through perturbations in:

1. **Initial seed location**  
2. **Graph topology**  
3. **Propagation dynamics**

---

## Data Sources

- Longitudinal and cross-sectional amyloid PET data  
- Alzheimer’s Disease Neuroimaging Initiative (ADNI)  
- Desikan-Killiany brain connectome atlas  

---

## Network Modeling Framework

### Propagation Models

- Watts Threshold Model (binary activation framework)  
- Diffusion-based propagation models  
- Prion-like spread simulations  

### Network Context Analysis

Regional vulnerability is assessed through:

- Degree centrality  
- Betweenness centrality  
- Clustering coefficient  
- Global efficiency  
- Community structure  

### Topological Data Analysis

To examine structural constraints and bottlenecks in disease spread:

- Persistent Homology  
- Persistent Representations  
- Bottleneck distances  

Topological features of the connectome are analyzed to understand propagation constraints and subtype differentiation.

---

## Subtype Integration (Future-planned)

Subtype staging derived from **SuStaIn** is used as an anchor to compare:

- Observed progression heterogeneity  
- Underlying network topology  
- Topological signatures of propagation  

This provides a principled way to link structural network architecture to disease staging variability.

---

## Key Hypothesis

Neurodegenerative disease trajectories can be modeled as:

> A propagation process shaped jointly by initial conditions, dynamical rules, and the topology of the brain’s structural network.

Understanding this interplay enables:

- Prediction of individual disease trajectories  
- Quantitative subtype characterization  
- Network-based risk assessment  
- Early identification of high-risk structural patterns  

---

# Methods & Tools

- Python (NetworkX, scientific computing stack)  
- Topological Data Analysis libraries  
- PET imaging data processing  
- Cox Proportional Hazards modeling  
- Longitudinal statistical modeling  
- Graph-theoretic analysis  

---

#  Applications

- Risk quantification in aging populations  
- Predictive modeling of AD progression  
- Network-informed intervention strategies  
- Structural vulnerability mapping  
- Mechanistic explanation of subtype heterogeneity  


#  Research Impact

This work reframes Alzheimer’s Disease as a **network-constrained propagation process**, integrating:

- Dynamical systems theory  
- Graph topology  
- Topological data analysis  
- Longitudinal neuroimaging  

By unifying these perspectives, this repository aims to contribute toward **mechanistic, predictive, and intervention-oriented models of neurodegeneration**.
