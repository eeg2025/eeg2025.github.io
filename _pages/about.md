---
layout: default
title: ""
permalink: /
subtitle: Transfer Learning, Domain Adaptation, Bio Signal, Brain Decoding, Electroencephalogram, Event-Related Potential

profile:
  align: right
  image:
  image_circular: false # crops the image to make it circular
  address: >

news: false # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false # includes social icons at the bottom of the page

bibliography: about.bib
---



<div align="center" style="margin: 1.5rem 0;">
  <h3 style="margin-bottom: 0.75rem;">
    Welcome Video for the EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding.
  </h3>
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/kSpnB_Rdo5c"
    title="Welcome Video for the EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding."
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    referrerpolicy="strict-origin-when-cross-origin"
    allowfullscreen
    loading="lazy"></iframe>
</div>

The 2025 EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding is a biosignal challenge accepted to the [**NeurIPS 2025 Competition Track**](https://neurips.cc/Conferences/2025/CallForCompetitions). This competition aims to advance the field of EEG decoding by addressing two critical challenges:

1. **Cross-Task Transfer Learning**: Developing models that can effectively transfer knowledge from passive EEG tasks to active tasks
2. **Subject Invariant Representation**: Creating robust representations that generalize across different subjects while predicting clinical factors

Check out the Challenge paper on arXiv: [10.48550/arXiv.2506.19141](https://arxiv.org/abs/2506.19141)

## Competition Tasks

<div style="padding: 20px; text-align: center;">
  <img alt="introduction" src="https://eeg2025.github.io/assets/img/workflow.png" style="max-width: 100%;" />
  <p style="text-align: left; margin-top: 10px; font-family: sans-serif; font-size: 0.9em">
    <b>Figure 1</b>: HBN-EEG Dataset and Data split. <b>A.</b> EEG is recorded using a 128-channel system during active tasks (i.e., with user input) or passive tasks. <b>B.</b> The psychopathology and demographic factors. <b>C.</b> The dataset split into Train, Test, and Validation. Details in <b><a href="https://eeg2025.github.io/assets/files/proposal.pdf">subsection 1.2 of the proposal.</a></b>
  </p>
</div>

### Challenge 1: Cross-Task Transfer Learning

This supervised learning challenge combines regression and classification objectives. Participants will predict behavioral performance metrics (response time via regression and success rate via classification) from an active experimental paradigm (Contrast Change Detection, CCD) using EEG data from a passive paradigm (Surround Suppression, SuS). Teams can leverage multiple datasets and experimental paradigms to train their models, utilizing unsupervised or self-supervised pretraining to capture latent EEG representations, then fine-tuning for the specific supervised objectives to achieve generalization across subjects and cognitive paradigms. See the [Starter Kit](baseline.md) for more details.

### Challenge 2: Psychopathology Factor Prediction (Subject Invariant Representation)

This supervised regression challenge requires teams to predict four continuous psychopathology scores (p-factor, internalizing, externalizing, and attention) from EEG recordings across multiple experimental paradigms. Teams can employ unsupervised or self-supervised pretraining strategies to learn generalizable neural representations, then adapt these foundation models for the regression targets while maintaining robustness across different subjects and experimental conditions. See the [Starter Kit](baseline.md) for more details.

## Dataset

The competition uses the HBN-EEG dataset ([paper](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2), [blog post](https://neuromechanist.github.io/data/hbn/)), which includes EEG recordings from over 3,000 participants across six distinct cognitive tasks:

### Passive Tasks

- **Resting State (RS)**: Eyes open/closed conditions with fixation cross
- **Surround Suppression (SuS)**: Four flashing peripheral disks with contrasting background
- **Movie Watching (MW)**: Four short films with different themes

### Active Tasks

- **Contrast Change Detection (CCD)**: Identifying dominant contrast in co-centric flickering grated disks
- **Sequence Learning (SL)**: Memorizing and reproducing sequences of flashed circles
- **Symbol Search (SyS)**: Computerized version of WISC-IV subtest

Each participant's data is accompanied by four psychopathology dimensions derived from the Child Behavior Checklist (CBCL) and demographic information, including age, sex, and handedness. Data is in the BIDS (Brain Imaging Data Structure) format.

## Awards

Top-ranking teams will be awarded with the following prizes:

- **Spotlight Talk** for the first team at the [2025 NeurIPS Workshop "Foundation Models for the Brain and Body"](https://brainbodyfm-workshop.github.io) to present their work (December 6-7, 2025)
- **Cash prizes** of 2500$ each for the top 3 teams (Sponsored by Meta)
- **Travel costs** covered for the main author of the top 3 teams (Sponsored by Meta)
- **Registration fees** for the whole NeurIPS 2025 conference covered for the main author of the top 3 teams

## Motivation

EEG decoding faces significant challenges due to signal heterogeneity from various factors like non-stationarity, noise sensitivity, inter-subject morphological differences, varying experimental paradigms, and differences in sensor placement. While recent advances in machine learning have shown promise, there remains a critical need for models that can generalize across different subjects and tasks without expensive recalibration.

This competition aims to address these challenges by:

1. Providing a large-scale, diverse dataset for evaluating model generalization
2. Encouraging the development of robust, transferable EEG decoding methods
3. Establishing benchmarks for cross-task and cross-subject performance
4. Fostering collaboration between machine learning and neuroscience communities

## Sponsors

<div style="padding: 20px; text-align: center;">
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-bottom: 20px;">
    <img src="https://eeg2025.github.io/assets/logos/meta.png" style="max-width: 50%; margin: 10px;" alt="Meta Logo" />
  </div>
</div>

## Organizing Institutions

<div style="padding: 20px; text-align: center;">
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-bottom: 20px;">
    <img src="https://eeg2025.github.io/assets/logos/lisn.png" style="max-width: 30%; margin: 10px;" alt="LISN Logo" />
    <img src="https://eeg2025.github.io/assets/logos/ucsd.png" style="max-width: 30%; margin: 10px;" alt="UCSD Logo" />
    <img src="https://eeg2025.github.io/assets/logos/donders.png" style="max-width: 30%; margin: 10px;" alt="Radboud University Logo" />
  </div>
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-bottom: 20px;">
    <img src="https://eeg2025.github.io/assets/logos/ups.png" style="max-width: 30%; margin: 10px;" alt="Université Paris-Saclay Logo" />
    <img src="https://eeg2025.github.io/assets/logos/inria.png" style="max-width: 30%; margin: 10px;" alt="Inria Logo" />
    <img src="https://eeg2025.github.io/assets/logos/icm.png" style="max-width: 30%; margin: 10px;" alt="Paris Brain Institute (Institut du Cerveau – ICM) Logo" />
  </div>
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-bottom: 20px;">
    <img src="https://eeg2025.github.io/assets/logos/chalearn.png" style="max-width: 30%; margin: 10px;" alt="ChaLearn Logo" />
    <img src="https://eeg2025.github.io/assets/logos/mind_intitute.png" style="max-width: 30%; margin: 10px;" alt="Child Mind Institute Logo" />
    <img src="https://eeg2025.github.io/assets/logos/nki.jpg" style="max-width: 30%; margin: 10px;" alt="Nathan Kline Institute for Psychiatric Research Logo" />
  </div>
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-bottom: 20px;">
    <img src="https://eeg2025.github.io/assets/logos/meta.png" style="max-width: 30%; margin: 10px;" alt="Meta Logo" />
    <img src="https://eeg2025.github.io/assets/logos/ben_logo.png" style="max-width: 30%; margin: 10px;" alt="Ben-Gurion University of the Negev Logo" />
    <img src="https://eeg2025.github.io/assets/logos/ufabc.png" style="max-width: 30%; margin: 10px;" alt="Federal University of ABC (UFABC) Logo" />
  </div>
  <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin-bottom: 20px;">
    <img src="https://eeg2025.github.io/assets/logos/cneuro.jpg" style="max-width: 10%; margin: 10px;" alt="Cuban Neuroscience Center (CNEURO) Logo" />
    <img src="https://eeg2025.github.io/assets/logos/UESTC.png" style="max-width: 30%; margin: 10px;" alt="University of Electronic Science and Technology of China Logo" />
    <img src="https://eeg2025.github.io/assets/logos/cnrs.png" style="max-width: 10%; margin: 10px;" alt="CNRS Centre National de la Recherche Scientifique Logo" />
    <img src="https://eeg2025.github.io/assets/logos/mcgill.png" style="max-width: 25%; margin: 10px;" alt="McGill University Logo" />
  </div>
</div>


## Core Coordinator

<div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap; margin-bottom: 20px;">
  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <a href="https://bruaristimunha.github.io/" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">
      <img src="https://eeg2025.github.io/assets/people/bruno.png" style="height: 100%; max-height: 135px; border-radius: 8px;" alt="Bruno Aristimunha" />
      <div style="font-weight: bold; margin-top: 10px;">Bruno Aristimunha</div>
    </a>
    <div style="font-size: 0.95em;">INRIA, University of Paris-Saclay</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <a href="https://neurotechlab.socsci.ru.nl/author/pierre-guetschel/" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">
      <img src="https://eeg2025.github.io/assets/people/pierre.jpeg" style="height: 100%; max-height: 135px; border-radius: 8px;" alt="Pierre Guetschel" />
      <div style="font-weight: bold; margin-top: 10px;">Pierre Guetschel</div>
      <div style="font-size: 0.95em;">Donders Institute, Radboud University</div>
    </a>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <a href="https://sylvchev.github.io/" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">
      <img src="https://eeg2025.github.io/assets/people/sylvain.jpg" style="height: 100%; max-height: 135px; border-radius: 8px;" alt="Sylvain Chevallier" />
      <div style="font-weight: bold; margin-top: 10px;">Sylvain Chevallier</div>
      <div style="font-size: 0.95em;">LISN, INRIA TAU, CNRS, A&amp;O Univ. Paris-Saclay</div>
    </a>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <a href="https://neuromechanist.github.io" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">
      <img src="https://eeg2025.github.io/assets/people/seyed.jpeg" style="height: 100%; max-height: 135px; border-radius: 8px;" alt="Seyed Yahya Shirazi" />
      <div style="font-weight: bold; margin-top: 10px;">Seyed Yahya Shirazi</div>
      <div style="font-size: 0.95em;">UC San Diego</div>
    </a>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <a href="https://arnauddelorme.com/" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">
      <img src="https://eeg2025.github.io/assets/people/arnaud.jpg" style="height: 100%; max-height: 135px; border-radius: 8px;" alt="Arnaud Delorme" />
      <div style="font-weight: bold; margin-top: 10px;">Arnaud Delorme</div>
      <div style="font-size: 0.95em;">CNRS, IONS, UCSD</div>
    </a>
  </div>
</div>

## Coordinators
<div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap; margin-bottom: 20px;">

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/alexandre_franco.png" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Alexandre R. Franco" />
    <div style="font-weight: bold; margin-top: 10px;">Alexandre R. Franco</div>
    <div style="font-size: 0.95em;">Child Mind Institute, USA</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/amit_majumdar.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Amit Majumdar" />
    <div style="font-weight: bold; margin-top: 10px;">Amit Majumdar</div>
    <div style="font-size: 0.95em;">UC San Diego, USA</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/aviv.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Aviv Dotan" />
    <div style="font-weight: bold; margin-top: 10px;">Aviv Dotan</div>
    <div style="font-size: 0.95em;">Ben-Gurion University of the Negev, Israel</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/dung.jpeg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Dung Truong" />
    <div style="font-weight: bold; margin-top: 10px;">Dung Truong</div>
    <div style="font-size: 0.95em;">ex-SCCN, UC San Diego, USA</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/evans_alan.png" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Alan Evans" />
    <div style="font-weight: bold; margin-top: 10px;">Alan Evans</div>
    <div style="font-size: 0.95em;">McGill University, Canada</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/gramfort.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Alexandre Gramfort" />
    <div style="font-weight: bold; margin-top: 10px;">Alexandre Gramfort</div>
    <div style="font-size: 0.95em;">Reality Labs, Meta, France</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/marie_constance_corsi.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Marie-Constance Corsi" />
    <div style="font-weight: bold; margin-top: 10px;">Marie-Constance Corsi</div>
    <div style="font-size: 0.95em;">Inria NERV; Paris Brain Institute (ICM), France</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/king.png" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Jean-Remi King" />
    <div style="font-weight: bold; margin-top: 10px;">Jean-Remi King</div>
    <div style="font-size: 0.95em;">FAIR Brain &amp; AI, Meta, France</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/milham.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Michael P. Milham" />
    <div style="font-weight: bold; margin-top: 10px;">Michael P. Milham</div>
    <div style="font-size: 0.95em;">Child Mind Institute, USA</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/oren.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Oren Shriki" />
    <div style="font-weight: bold; margin-top: 10px;">Oren Shriki</div>
    <div style="font-size: 0.95em;">Ben-Gurion University of the Negev, Israel</div>
  </div>
  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/scott_makeig.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Scott Makeig" />
    <div style="font-weight: bold; margin-top: 10px;">Scott Makeig</div>
    <div style="font-size: 0.95em;">SCCN; Institute for Neural Computation, UC San Diego, USA</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/pedro_valdes_sosa.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Pedro A. Valdés-Sosa" />
    <div style="font-weight: bold; margin-top: 10px;">Pedro A. Valdés-Sosa</div>
    <div style="font-size: 0.95em;">Cuban Neuroscience Center (CNEURO); Joint China-Cuba Lab (UESTC)</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/terrence_sejnowski.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Terrence J. Sejnowski" />
    <div style="font-weight: bold; margin-top: 10px;">Terrence J. Sejnowski</div>
    <div style="font-size: 0.95em;">UC San Diego; Salk Institute, USA</div>
  </div>

  <div style="max-width: 160px; text-align: center; margin: 10px;">
    <img src="https://eeg2025.github.io/assets/people-h180/isabelle.jpg" style="height: 100%; max-height: 180px; border-radius: 8px;" alt="Isabelle Guyon" />
    <div style="font-weight: bold; margin-top: 10px;">Isabelle Guyon</div>
    <div style="font-size: 0.95em;">Google DeepMind; ChaLearn, USA</div>
  </div>
  
</div>
