# NatureCommunications
This repository contains analysis code for the following paper:  Rollwage, et al. (2020) Confidence drives a neural confirmation bias. Nature Communications

Fully anonymised data files are available from the corresponding author on reasonable request (max.rollwage.16@ucl.ac.uk). Scripts for analysing the behavioral data (factor analysis), drift-diffusion modelling and MEG analysis are included in the repository:

The script Behavioral_analysis_Study1.m reproduces the behavioral analysis of study 1 regarding the influence of confidence on changes of mind (Figure 1).

The script HDDM_model_fit.py fits the different drift-diffusion models and evaluates the posterior distribution of models paramters (Figure 2D).

The script HDDM_model_simulation.m simulates the best fitting drift-diffusion model (Figure 2B&C).

The script Neural_evidence_accumulation.m analysis and validates the neural measures of evidence accumulation (Figure 3B, D-F)
and shows how neural post-decision evidence accumulation is biased by previous confidence and decision (Figure 4A-C)

The script Temporal_Generalization.m investigates how the time course of neural representations of an initial decision is modulated by confidence during the post-decision period (Figure 4D)
