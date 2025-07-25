---
layout: page
permalink: /baseline/
title: Starter&nbsp;Kit
description:  
nav: true
nav_order: 5

---

The starter kit will be released following the timeline details to give equal opportunities to people who cannot download the dataset and will depend on our infrastructure to run the code.

This code relies on the [Braindecode](https://braindecode.org) and [EEGDash](https://eegdash.org) libraries. These libraries allow data search, loading, fetching, and readily applying deep learning methods to EEG data. The provided scripts are only for reference, and contestants can freely use their own codebase for development.

## Challenge 1: Cross-Task Transfer Learning Data

For Challenge 1, participants will have access to the following data for validation and test phases:

**EEG Data:**
- **Surround Suppression (SuS) task**: Complete EEG recordings from the passive paradigm
- **Pre-trial epochs**: 2-second EEG data segments preceding each contrast change event (see figure below)
- **Recording specifications**: 128-channel high-density EEG at standard sampling rate

**Additional Features:**
- **Demographics**: Age, sex (gender), and handedness (Edinburgh Handedness Questionnaire scores)
- **Psychopathology factors**: Four bifactor model dimensions (p-factor, internalizing, externalizing, attention)

**Target Variables** (validation only):
- Response time relative to contrast change onset (regression target)
- Success rate/hit accuracy (classification target)

Formally, let $$X_1 \in \mathbb{R}^{c \times n \times t_1}$$ denote a participant's EEG recording during the CCD task (total 3 runs), where $$c = 128$$ channels, $$n \approx 70$$ epochs, and $$t_1 = 2$$ seconds (epoch length). Let $$X_2 \in \mathbb{R}^{c \times t_2}$$ represent a participant's EEG recording during the SuS task (total 2 runs), where $$c = 128$$ channels and $$t_2$$ is the total number of time samples. Let $$P \in \mathbb{R}^7$$ be the subject's traits, including 3 demographic attributes and 4 psychological factors. Participants SHOULD use $$X_1$$ but can choose to use $$X_2$$ and $$P$$ as additional data modalities and features to train their models or to use at inference time.

<div style="text-align: center; margin: 20px 0;">
  <img src="https://eeg2025.github.io/assets/img/CCD_sequence.png" alt="CCD Trial Sequence" style="max-width: 80%; height: auto;">
  <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
    <strong>Figure:</strong> Exemplar Contrast Change Detection (CCD) trial sequence showing the 2-second pre-trial epochs (hatched regions) that will be provided as input data for Challenge 1.
  </p>
</div>

## Challenge 2: Psychopathology Factor Prediction Data

For Challenge 2, participants will have access to the following data for validation and test phases:

**EEG Data:**
- **Multi-task recordings**: EEG data from all available cognitive paradigms (passive and active tasks)
- **Minimum data requirement**: Only subjects with ≥15 minutes of total EEG data included (>78% of participants)
- **Recording specifications**: 128-channel high-density EEG across multiple experimental sessions

**Input Features:**
- **Demographics**: Age, sex (gender), and handedness (Edinburgh Handedness Questionnaire scores)
- **EEG recordings**: Complete datasets from multiple cognitive tasks per subject

**Target Variables** (validation only):
- **P-factor**: General psychopathology factor (continuous score)
- **Internalizing**: Inward-focused traits dimension (continuous score) 
- **Externalizing**: Outward-directed traits dimension (continuous score)
- **Attention**: Focus and distractibility dimension (continuous score)

All psychopathology factors are derived from Child Behavior Checklist (CBCL) responses using a bifactor model, providing orthogonal, privacy-preserving dimensional measures of mental health.

Formally, let $$X \in \mathbb{R}^{c \times t}$$ denote a participant's EEG recording across multiple cognitive tasks, where $$c = 128$$ channels and $$t$$ is the total number of time samples. Let $$D \in \mathbb{R}^3$$ represent the subject's demographics, including age, sex (gender), and handedness. Participants should use $$X$$ and potentially $$D$$ to train their models for predicting the target variables $$P \in \mathbb{R}^4$$, including p-factor, internalizing, externalizing, and attention.

### Task events
Most tasks also include rich event information, including start and stop of stimulus presentation, responses, and other task-specific events annotated in HED (Hierarchical Event Descriptor) format. Participants can use this information to train their models or use it at inference time for both challenges.

---

**Note:** Starter kit code samples and baseline model implementations will be added following the competition timeline. Participants are encouraged to explore the data structure and develop their approaches using the validation set during the warm-up phase.
