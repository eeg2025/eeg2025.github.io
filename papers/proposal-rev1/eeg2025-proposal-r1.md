# 2506.19141v2.pdf

# EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding 

Bruno Aristimunha ${ }^{* 1 \text { a }}$ Dung Truong ${ }^{\ddagger}$ Pierre Guetschel ${ }^{\S}$ Seyed Yahya Shirazi ${ }^{\ddagger}$<br>Isabelle Guyon ${ }^{\text {e }}$ Alexandre R. Franco ${ }^{\S}$ Michael P. Milham ${ }^{\S}$ Aviv Dotan ${ }^{\text {A }}$<br>Scott Makeig ${ }^{\ddagger}$ Alexandre Gramfort ${ }^{\text {® }}$ Jean-Remi King ${ }^{\text {® }}$ Marie-Constance Corsi ${ }^{\dagger}$<br>Pedro A. Valdés-Sosa ${ }^{\ominus}$ Amit Majumdar ${ }^{\S}$ Alan Evans ${ }^{\ominus}$ Terrence J Sejnowski ${ }^{\ddagger}$<br>Oren Shriki ${ }^{\text {A }}$ Sylvain Chevallier*<br>Arnaud Delorme ${ }^{\text {I }}$ (1)<br>neurips2025-eeg-competition@googlegroups.com


#### Abstract

Current electroencephalogram (EEG) decoding models are typically trained on small numbers of subjects performing a single task. Here, we introduce a largescale, code-submission-based competition comprising two challenges. First, the Transfer Challenge asks participants to build and test a model that can zero-shot decode new tasks and new subjects from their EEG data. Second, The Psychopathology factor prediction Challenge asks participants to infer subject measures of mental health from EEG data. For this, we use an unprecedented, multi-terabyte dataset of high-density EEG signals ( 128 channels) recorded from over 3,000 child to young adult subjects engaged in multiple active and passive tasks. We provide several tunable neural network baselines for each of these two challenges, including a simple network and demographic-based regression models. Developing models that generalize across tasks and individuals will pave the way for ML network architectures capable of adapting to EEG data collected from diverse tasks and individuals. Similarly, predicting mental health-relevant personality trait values from EEG might identify objective biomarkers useful for clinical diagnosis and design of personalized treatment for psychological conditions. Ultimately, the advances spurred by this challenge could contribute to the development of computational psychiatry and useful neurotechnology, and contribute to breakthroughs in both fundamental neuroscience and applied clinical research.


Keywords Transfer Learning, Biosignal, Brain Decoding, Electroencephalogram, Time Series

## 1 Competition description

We propose the EEG Foundation Challenge to advance research on developing generalized and transferable neural representations for EEG decoding (Figure 1). This initiative focuses on identifying mental biomarkers and decoding cognitive tasks across large number of subjects, addressing the fundamental challenge of interpreting complex brain signals associated with key psychopathological factors and experimental stimuli.

[^0][^1]
[^0]:    * A\&O, LISN, Université Paris-Saclay, CNRS, Inria TAU, France ${ }^{\dagger}$ Inria NERV, Paris Brain Institute, France
    ${ }^{\ddagger}$ SCCN, INC, SDSC, University of California San Diego, USA ${ }^{\ominus}$ CNRS, France
    ${ }^{\S}$ Donders Institute, Radboud University, Netherlands
    ${ }^{\S}$ Child Mind/Nathan Kline Institutes, New York, USA
    ${ }^{\text {® }}$ FAIR Brain \& AI team/Reality Labs, Meta, France
    ${ }^{\ominus}$ Cuban Neuroscience Center, Cuba and China

    ${ }^{\text {® }}$ Google DeepMind, ChaLearn, USA
    ${ }^{\text {A }}$ Ben-Gurion University of the Negev, Israel
    ${ }^{\text {A }}$ Federal University of ABC, Brazil
    ${ }^{\ominus}$ McGill University, Canada

[^1]:    Preprint. Under review.

![img-0.jpeg](images/img-0.jpeg)

Figure 1: HBN-EEG Dataset and Data split. **A.** EEG is recorded using a 128-channel system during active tasks (i.e., with user input) or passive tasks. **B.** The psychopathology and demographic factors. **C.** The datasets split into Train, Test and Validation. Details in subsection 1.2.

In recent years, the neuroscience community has increasingly looked toward Large Language-Model-style architectures to advance brain activity decoding [57]. However, this paradigm shift remains limited by the data scale and structure of currently available data [8, 22]. Unlike the massive, richly annotated corpora used to train language models, EEG datasets are relatively small and lack comparable hierarchical organization. Moreover, brain signal data present intrinsic challenges, including differences in electrode montages, recording protocols, various methods for splitting the data, and underlying neurological factors [6, 11, 17, 31, 51]. These challenges have hindered the development of broad, generalizable, and cross-task models that extend beyond simple classification, highlighting the need for larger, more integrated EEG datasets to support next-generation research.

A given EEG dataset usually corresponds to one specific cognitive task or condition with only a single type of ground-truth label, often requiring a separate model to be trained for each task. In practice, models are frequently developed and tuned in isolation on each dataset—and sometimes even on each individual subject—due to pronounced physiological differences between subjects [3, 36, 44]. This conventional single-task, -subject training and evaluation fails to leverage commonalities across datasets and yields models that struggle to generalize beyond the context in which they were trained.

For these reasons, we argue that a general-purpose EEG foundation competition is needed to move beyond the one-task-per-model paradigm. A unified model could simultaneously decode both enduring mental traits and dynamic cognitive states, bridging traditionally separate objectives. For instance, it could infer a participant's latent mental health score while also predicting task performance from the same EEG stream. Training on diverse tasks and subjects would enable the model to learn robust representations that generalize across users and adapt to different electrode configurations. As model capacity and data diversity scale, we expect more reliable regression of cognitive and clinical biomarkers, even in the presence of noise.

Previous EEG decoding challenges have shown that well-designed competitions can drive progress in brain decoding. Our BEETL Motor Imagery Challenge at NeurIPS 2021 [54] gathered 130+ contestants from 40 research groups and over 1,382 submissions. Our Brain Age Prediction Challenge 2022 [37] attracted 200+ participants from 40 countries, generating 30–100 daily submissions. In 2023, our Sleep States Competition [26] engaged 80 teams from 20 countries. In 2024, we co-organized a large-scale challenge with the Child Mind Institute [46], focusing on predicting problematic internet use in youth using accelerometry, assessments, and questionnaires. It drew 12,912 registrants, 2,436 active contestants across 1,877 teams, and 38,329 submissions. This year,

we are currently running additional challenges on the Healthy Brain Network's fMRI and actigraphy. The ongoing WiDS Datathon 2025 [10], part of the Women in Data Science initiative, has already reached 5,062 contestants, with 1,614 active contestants across 703 teams and 9,470 submissions to date. These results highlight our strong track record of organizing impactful, large-scale challenges and mobilizing the machine learning research community for an appealing NeurIPS competition.

# 1.1 Novelty 

The EEG Foundation Challenge is inspired by and builds up on our NeurIPS 2021 BEETL challenge, with a bigger scope and ambition. Unlike BEETL, which focuses on specific tasks (sleep staging and motor imagery), the present competition introduces an unprecedented combination of scale, complexity, and objectives. The competition features three major innovations:

1) Unprecedented Scale and Complexity: The competition's dataset encompasses over 3,000 participants with high-density 128-channel EEG recordings - an order of magnitude larger than typical EEG challenge datasets. Each participant engaged in six distinct cognitive tasks (ranging from resting-state to active learning and attention tasks), providing a rich multi-task, multi-condition collection of neural data. This participant, metadata, and recording breadth and diversity far exceeds that of any prior EEG competition [26, 54, 37]. So, models should handle heterogeneous time-series data conditions, learn generalizable representations across tasks and participants. EEG decoding research has progressed beyond traditional discrete classification methods, with novel approaches now addressing more complex representational challenges at this large scale [57, 8, 7, 53, 52, 32, 16, 23].
2) Zero-Shot Cross-Domain Generalization: The cross-task transfer learning scenario in EEG decoding is remarkably underexplored [5]. Our Challenge 1 addresses a key goal in neurotechnology: decoding cognitive function from EEG without explicit behavioural labels. Participants must develop models capable of zero-shot generalization across both novel tasks and novel subjects. The competition is structured to reward models that learn domain-invariant and subject-invariant representations. This means a submitted model might be trained on a subset of tasks and then tested on data from a held-out task or condition, evaluating its capacity to generalize without task-specific fine-tuning. Notably, no existing benchmarks or datasets have been developed to address these issues [14, 20].
3) Prediction of Latent Psychological Constructs: In contrast to past challenges, our Challenge 2 targets latent psychological constructs as prediction outputs. Specifically, participants will predict transdiagnostic psychopathology factors (such as the p-factor) derived from standardized clinical questionnaires [48, 12]. While the p-factor is intended to reflect a psychological trait [12], EEG could also be modulated by fluctuating "states" such as engagement, fatigue, and other time-variant components [9, 43]. These components may be implicitly encoded in neural activity, helping to account for such variabilities in the data. The challenge holds considerable potential for scientific impact, particularly if participating teams could improve decoding of psychopathology factors from EEG data, thereby offering empirical support for EEG-based biomarkers for mental health [27].
By combining unexplored targets, fundamental scientific questions, and a large-scale, highdimensional dataset, this competition is a unique opportunity to advance the state of EEG-based predictive modelling and to use machine learning in the discovery of psychiatric biomarkers. Beyond its methodological and scientific contributions, this challenge has meaningful societal implications: identification of psychopathology metrics through accessible, non-invasive measures like EEG could contribute to earlier detection, more precise monitoring, and ultimately more effective interventions for mental health conditions. As mental health continues to emerge as a global public health priority, innovations in scalable and data-driven assessment methods can inform clinical practice and policy, reduce stigma, and improve outcomes for individuals affected by psychiatric disorders.

### 1.2 Data

The competition leverages the Healthy Brain Network Electroencephalography (HBN-EEG) datasets [49]. It is made of 12 dataset releases in total, grouped according to the time of release, 11 releases shared, and one withheld and unreleased for competitions. This large-scale collection contains high-density (128-channel) EEG recordings from children and young adults aged 5-21 years with more details described at [49]. The dataset is formatted according to the Brain Imaging Data Structure (BIDS) standard [24, 40] and includes comprehensive event annotations using Hierarchical

Event Descriptors (HED) [42, 35], making it particularly suitable for cross-task analysis and machine learning applications.
Formally, let $X \in \mathbb{R}^{c \times t}$ be an EEG recording segment (Figure 1), with $c=128$ the number of electrodes and $t$ the time steps, $Y \in \mathbb{R}^{k \times t}$ be the associated labels, with $k$ the number of event labels, and $P \in \mathbb{R}^{7}$ be subject's traits including 3 demographic attributes and 4 psychological factors, e.g., p-factor $\mathbf{p}_{f a c}$. A dataset is defined as $(\mathcal{X}, \mathcal{Y}, \mathcal{P})$, with $N$ EEG recording from $S$ subjects, $\mathcal{X}=\left\{\mathrm{X}_{i}\right\}_{N}$ with their associated labels $\mathcal{Y}=\left\{\mathcal{Y}_{i}\right\}_{N}$ and psychological factors and demographic attributes $\mathcal{P}=\{\mathrm{p}\}_{7}$ for each $s$ subject.
We consider here different datasets with different types of labels. In this competition, the trial timing is assumed to be known, targeting offline applications that are common in the medical context, leaving real-time decoders outside the scope of this competition. We assume that the dataset is randomly split into a training set of size $N_{\text {train }}$, a validation set of size $N_{\text {valid }}$ and a test set of size $N_{\text {test }}$. For training and validation sets, both $\mathcal{X}$ and $\mathcal{Y}$ are available, while for test sets, only $\mathcal{X}$ is available to the contestant. The general decoding problem amounts to using the training set $\mathcal{X}_{\text {train }}$ to learn a model $f_{\theta}$ with parameters $\theta$ mapping each trial X to the associated label Y .

# 1.2.1 Dataset Description 

The competition data set includes EEG recordings from more than 3, 000 participants in six distinct cognitive tasks, divided into passive - Resting State (RS), Surround Suppression (SuS), Movie Watching (MW) - and active - Contrast Change Detection (CCD), Sequence Learning (SL), Symbol Search (SyS) - categories. SuS and CDD rely on stimulus that are similar to those employed in SSVEP tasks.
The data of each participant are accompanied by four dimensions of psychopathology (internalization, externalization, attention, and p-factor) derived from a bifactor model of parent-reported questionnaire responses to the Child Behavior Checklist (CBCL) [1, 48]. These psychopathology factors represent orthogonal dimensions of mental health and serve as target variables for the regression component for Challenge 2 [48]. Additionally, the age, gender, and handedness (scored using the Edinburg Inventory, EHQ-Total) [38] are included as part of the demographic information for each participant. While some psychopathology factors, like internalizing and externalizing, are influenced by a participant's age, other factors such as the p-factor and attention, are largely unaffected by age Figure 1B.

Data Access and Ethics The first 11 releases of the HBN-EEG dataset are freely available under CC-BY-SA-4.0 licence through NEMAR [18]. Testing set use release 12 that is currently unreleased. The Chesapeake Institutional Review Board approved the data collection, with written informed consent obtained from participants 18 years or older and from legal guardians of younger participants. All participants were anonymized using the NIMH's Global Unique Identifier Tool (GUID) [30] to ensure no exposure to personally identifiable information.

### 1.2.2 Competition Data Split

For this competition, we provide contestants with a Training set, complete EEG recordings and psychopathology measures from approximately 3,000 subjects, releases 1-11, except for the Release 5. The Validation set, Release 5 comprises data from the remaining subjects for the teams to test their models. This split ensures sufficient data for model training while maintaining a robust evaluation set for assessing cross-task generalization and psychopathology prediction performance. The final evaluation will be performed using the withheld test set, the Release 12 that is not available publicly, but will be released publicly in the foreseeable future after the competition.
As contestants submit their code, they will not have direct access to the test set. No subject's data is spread across multiple releases; this means the Training, Validation, and Testing are completely separate in terms of subjects, ensuring no feature leakage in this competition, avoiding subjects' co-founders. An illustration of the data split can be found in Figure 1.
For the Challenges, participants are strongly encouraged to leverage additional datasets beyond the provided training set, particularly those involving SSVEP stimuli. As a starting point, we recommend exploring the Mother of All BCI Benchmarks (MOABB) library [14, 4], which provides a wide range of small, but well-curated EEG datasets. However, contestants are responsible for designing models capable of adapting to variations in channel configurations and signal characteristics across datasets.

# 1.3 Tasks and application scenarios 

The competition focuses on two primary challenges that address significant problems in neurophysiological research and clinical neuroscience: inferring mental states and inferring mental traits from EEG signals. From a machine learning perspective, the model needs to generalize across two challenges: a transfer learning problem and a regression problem. In Challenge 1, the source domain is the concatenation of multiple datasets with multiple tasks on train subjects, and the target domain is a specific task on test subjects, whereas in Challenge 2, the objective is to predict a specific psychological trait from EEG recording without labels and with the other subject's traits:

$$
\text { Challenge } 1-f_{\theta}:(\mathcal{X}, \mathcal{P}) \rightarrow \mathcal{Y}, \quad \text { Challenge } 2-f_{\theta}:(\mathcal{X}, \mathcal{Y}) \rightarrow \mathcal{P}
$$

In Challenge 1 - Transferability across subjects and cognitive tasks, the contestants are provided with multiple databases with subjects performing similar tasks, namely SSVEP with ERP components, to train their model on the HBN Releases datasets. For this challenge, the contestants must predict behavioural performance metrics in an active task (Contrast Change Detection, CCD): response time relative to the start of contrast change index on the stimulus time with a 500 ms shift, i.e., The window period is 0.5 to 2.5 seconds after stimulus onset for prediction.

At the inference time, we will use the Contrast Change Detection data with the two-second data epochs and plus their demographic at the epoch level, as indicated in Figure 1.

Detailed data split, epoching, and basic scripts for data loading and curation are available at the competition website. Contestants should submit per-trial predictions. The metrics used to evaluate the predictions are detailed in Sect 1.4.

This challenge has significant implications for clinical and research settings as obtaining and retaining EEG recordings and accounting for participant's response variability may be difficult or impossible due to participants' or environment limitations [21, 19]. The experiment length and actively engaging participants in experiments are challenging for several populations, including children, individuals with severe motor impairments, or those with communication and developmental disorders [21, 56, 41]. Passive EEG paradigms such as movie watching with shorter duration that can predict performance capabilities would enable more inclusive and accessible neurological assessments. Additionally, by extracting maximal information from minimal recording sessions, researchers can reduce the time and effort required from participants, making neurophysiological research more accessible and efficient.

In Challenge 2 - Decoding psychopathology factor from the EEG, the contestants will have to regress the continuous four psychopathology scores using EEG recordings from multiple cognitive tasks. For this challenge, only subjects with at least 15 minutes of total EEG data and complete demographic information will be included for validation and test ( $>78 \%$ of the participants). Contestants should predict per-subject score. The metrics used to evaluate the predictions are detailed in section 1.4.

This challenge represents a novel approach to mental health assessment that leverages objective neurophysiological measures instead of relying on subjective self-reports or clinical observations. It addresses critical challenges in mental health assessment and treatment. Current psychiatric diagnoses mainly rely on subjective assessments, leading to potential inconsistencies and biases [34]. EEGbased biomarkers could provide complementary, physiologically-grounded measures of mental health [9]. Neurophysiological signatures of psychopathology may be detectable before clinical symptoms fully manifest, potentially enabling earlier intervention. By predicting continuous psychopathology dimensions rather than categorical diagnoses, this approach aligns with modern transdiagnostic frameworks like the Research Domain Criteria that conceptualizes mental health along continuous dimensions [28, 15]. Also, objective EEG-based measures could be used as biomarkers to track treatment response over time, providing quantitative feedback on intervention efficacy.

Scientific and Technical Challenges: Both challenges raise substantial scientific and technical questions that are central to the development of EEG foundation models. EEG signals are inherently high-dimensional, noisy, and temporally complex, with considerable inter-subject and inter-task variability. Building models that generalize across subjects, tasks, and recording (electrode) configurations requires scalable architectures capable of robust pattern recognition in the face of such variability.

# 1.3.1 Ethical Considerations 

To consider the potential ethical implications of these challenges, we implemented the following steps: 1) We ensured that all data have collected with appropriate consent. 2) We focus here on psychological traits instead of diagnostic categories, to mitigate the risks of diagnostic misuse and to limit the potential for discriminatory use of EEG-based predictions. 3) We aim to develop models that perform equitably across different populations by using a diverse dataset and including various demographic factors.

The development of highly accurate, task- and subject-general models has the potential to improve the identification of interpretable neural biomarkers, thereby supporting clinical insight and decision-making by health professionals. The competition challenges align with the NeurIPS community's interests in developing robust, generalizable machine learning methods for complex, high-dimensional time series data while addressing significant real-world challenges in healthcare and cognitive science.

### 1.4 Metrics

We employ carefully selected metrics to evaluate performance on each challenge, ensuring they reflect real-world utility and statistical robustness.

Challenge 1 Contestants will predict behavioural performance, i.e., the response time, in the Contrast Change Detection (CCD) task using EEG data from the Surround Suppression (SuS) task and pre-trial EEG. We decided not to use correct/incorrect classifications for this challenge. The evaluation metric for this regression challenge is the normalized root mean square error $\operatorname{nRMSE}(\cdot, \cdot)$ between the true and predicted response time. The normalized root mean square error is the metric quantifying prediction error magnitude, providing a clinically interpretable measure of prediction deviation Formally, this metric is defined as follows:

$$
\begin{aligned}
\operatorname{RMSE}\left(\mathbf{Y}_{\text {true }}, \mathbf{Y}_{\text {pred }}\right) & =\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left|\mathbf{Y}_{\text {true }}^{i}-\mathbf{Y}_{\text {pred }}^{i}\right|^{2}} \\
\operatorname{std}(\mathbf{Y}) & =\sqrt{\frac{1}{N} \sum_{i=1}^{N}\left|\mathbf{Y}^{i}-\overline{\mathbf{Y}}\right|^{2}} \\
\operatorname{nRMSE}\left(\mathbf{Y}_{\text {true }}, \mathbf{Y}_{\text {pred }}\right) & =\frac{\operatorname{RMSE}\left(\mathbf{Y}_{\text {true }}, \mathbf{Y}_{\text {pred }}\right)}{\operatorname{std}\left(\mathbf{Y}_{\text {true }}\right)} \\
\mathcal{S}_{1} & =\operatorname{nRMSE}\left(\mathbf{Y}_{1, \text { true }}, \mathbf{Y}_{1, \text { pred }}\right)
\end{aligned}
$$

With $\mathbf{Y}_{1, \text { true }}$ being the vector containing the true response times, $\mathbf{Y}_{1, \text { pred }}$ the predicted ones, and $\mathcal{S}_{1}$ the score for challenge 1.

Challenge 2 Contestants will develop models to predict the psychopathology factors from EEG recordings. The evaluation metric for this challenge is also the normalized root mean square error:

$$
\mathcal{S}_{2}=\operatorname{nRMSE}\left(\mathbf{Y}_{2, \text { true }}, \mathbf{Y}_{2, \text { pred }}\right)
$$

With $\mathbf{Y}_{2, \text { true }}$ being the labels vector containing the p-factors obtained from questionaries, $\mathbf{Y}_{2, \text { pred }}$ containing the p -factors predicted from the EEG recordings, and $\mathcal{S}_{2}$ the score for challenge 2.

Overall Ranking The objective of this competition being to foster the development of foundation models, the overall ranking of this competition is a combination of both challenges to promote the design of models able to address multiple tasks. The final score $\mathcal{S}_{\text {overall }}$ reflects the greater clinical significance of psychopathology prediction. This comprehensive evaluation framework ensures that winning approaches demonstrate robust performance across multiple clinically relevant dimensions, encouraging the development of models with real-world utility for neurophysiological assessment and mental health applications.

$$
\mathcal{S}_{\text {overall }}=0.3 \mathcal{S}_{1}+0.7 \mathcal{S}_{2}
$$

# 1.5 Baselines, code, and material provided 

The starter kit can be found at: https://eeg2025.github.io/baseline/. This code relies on the BRAINDECODE and EEGDASH libraries. These libraries allow data search, loading, fetching, and readily applying deep learning methods on EEG data. In particular, BraindeCODE bridges the MNEPython [25] and PyTorch [39] libraries (respectively made for brain signal processing and deep learning). EEGDASH allows to train BraindeCode models or any PyTorch model from data retrieved from OpenNeuro and NEMAR. In the BraindeCode library, we provide a PyTorch model zoo for EEG decoding featuring over 30 deep learning models-including ShallowConvNet [47], EEGNet [33], EEG-Inception [45], EEG-Conformer [50], EEGNex [13], ATCNet [2], BIOT [55], Labram [29] and more, all rigorously standardized and tested for correctness. The provided scripts are only for reference, and the contestants can freely use their codebase for development.

### 1.6 Website, tutorial and documentation

All information regarding this competition will be provided at https://eeg2025.github.io. This website will contain a description of the data, links and command lines to download the data, a script allowing the load the data and run the baseline(s), the timeline, the leaderboard, FAQ, and news. We also provide a dedicated e-mail address for communication with contestants. We will create a GitHub Community Forum for the competition to facilitate communication between participants and organizers, as well as among participants themselves.

## 2 Organizational aspects

### 2.1 Protocol

Contestants are provided will all necessary scripts to download the data. Some model examples are provided, but they could choose freely their model. They need to train their model on their own computational infrastructure, we provide example how to use Google Colab for contestant without specific resources. They need to create a team account on the CODABENCH platform and to provide a code submission with their trained model. The Université Paris-Saclay and San Diego Supercomputer Center will provide the compute worker nodes to run the model against test.

The competition will have two phases: in the warm-up phase, the contestants start to submit their code, the evaluation will be performed on validation set (HBN Release 5) that is publicly available, allowing contestants to verify that their code is running correctly. No limit will be fixed to the number of submissions to facilitate debugging. This phase should serve as an incentive to incite contestants to join the competition, with an easy to beat baseline and several available models, ready to be tuned.

During the second final phase, the contestants' model will be evaluated on test set (HBN Release 12, not publicly available). The number of submissions per day will be limited to avoid overfitting the leaderboard. Their final rankings and scores will be released after the competition deadline. Each team will upload a 2-page document, in paper format with methods, analysis and discussion.

### 2.2 Rules and Engagement

The rules that will be provided to the contestants are the following:

- Contestants are allowed and encouraged to use any datasets to pre-train.
- Contestant submit their code during the inference stage, this is a code submission competition.
- Related members from the organising team can participate, but are ineligible for prizes.
- The top 10th teams will have their code release after the final submission.


### 2.3 Schedule and readiness

The tentative timeline for challenge preparation is:

- [done] Finishing dataset interface with OPENNEURO and NEMAR portals.
- [done] Finishing baseline integration using BraindeCode models.
- [done] Verifying the reproducibility of the scripts.
- 30/04/2025: Cleaning up the script for releasing.

- 17/05/2025: Extend CodaBench configuration to run on NSF's ACCESS or bigger cluster.
- 18/05/2025: Complete internal beta testing to identify and address potential bottlenecks.
- 18/05/2025: Announcement of the challenge on social media and our dissemination channels as describe at subsection 2.4.

The proposed timeline for the challenge is as follows:

- 15/07/2025: Warm-up Phase started.
- 15/09/2025: Final Phase started with unreleased data.
- 31/10/2025: End of the competition.
- 30/11/2025: Competition reports and competition analysis paper.
- At NeurIPS: Organization of the competition session, with keynote about the lessons learned.


# 2.4 Competition promotion and incentives 

We anticipate participation from three key groups: (1) researchers with proven success in EEG decoding, particularly transfer learning or psychopathology detection; (2) researchers seeking to benchmark strong, unpublished approaches on a high-visibility platform like NeurIPS; and (3) machine learning experts, especially from transfer learning domains, interested in applying their methods to EEG data.

To reach these communities, we will directly contact over 4,000 participants from our five previous decoding competitions and disseminate the call via major mailing lists (including EEGLAB with 17,000+ members, MNE-Python, and NeuroTechX). We will further leverage the organizing team's extensive professional networks across academia and industry for broad promotion. To incentivize strong participation, significant awards are offered:

- The top three teams will each receive a USD 2,500 cash prize and full travel support (covering transportation, accommodation, and registration) for one representative to present at NeurIPS.
- The top five teams will be invited to present their work during the competition track (15-minute slots each) and will be invited to contribute to a special issue of a leading journal (TBC).
- The top three teams will also be recognized as consortium authors on the subsequent NeurIPS competition publication, following the format of our previous NeurIPS competition [54].

To actively encourage newcomers and researchers from under-represented backgrounds, we will offer a dedicated prize of USD 1,000 and NeurIPS workshop registration to the best-performing first-time team led and with the majority of women or members of minority groups, adhering to NeurIPS DIA guidelines. We are committed to inclusivity and will partner with advocacy groups such as Women in AI, Black in AI, Queer in AI, and LatinX in AI to broaden our outreach. Finally, hosting a forum will foster continued engagement beyond the competition, promoting code sharing, collaboration and discussions on EEG decoding to foster collaboration and reproducible research [11].

## 3 Resources

3.1. Organizing team The organization is diverse and includes scientists from different institutions and countries, including France, USA, Israel, Canada, Cuba, China, Brazil and Netherlands. The individual contributions are listed below and more details on the team can be found in Appendix A.

- Coordinators Bruno, Dung, Pierre, and Yahya serve as core coordinators. Isabelle, Sylvain, and Alexandre F. provide strategic oversight. Isabelle and the ChaLearn team support Codalab hosting. Amit provides access to SDSC compute resources.
- Data Providers: Michael M. and Alexandre F. provide the EEG datasets. Pedro and Alan: datasharing expertise. Yahya, Scott, Oren, Amit, and Arnaud: BIDS, HED, and NEMAR expertise.
- Baseline Method Providers: Bruno and Pierre (deep learning). Aviv (machine learning for computational psychiatry). Dung and Arnaud (HBN data).
- Beta Testers: Jean-Rémi with his team his team, Yahya and Marie-Constance assist in early testing and validation of the challenge platform.
- Evaluators: Arnaud, Sylvain, Alexandre G., Isabelle Guyon and Terry Sejnowski contribute domain-specific evaluation, ensuring scientific rigor.

3.2. Resources provided by organizers: We provide HPC resources with GPUs through Mesocentre Paris-Saclay, which were used to evaluate submissions for the preliminary edition of the competition at CODABENCH.COM. These could be used for conducting during the inference phase.
3.3. Support requested: We would greatly appreciate support from the NeurIPS 2025, particularly in promoting our challenge through their channels.

# A Biography of all team members 

## 1. Bruno Aristimunha

- Affiliation: A\&O, LISN, Université Paris-Saclay, CNRS, Inria TAU, France; Inria NERV, Paris Brain Institute, France; Federal University of ABC, Brazil
- E-mail: b.aristimunha@gmail.com
- Research Engineer at INRIA TAU and PhD student at the University of Paris-Saclay.
- Machine Learning Research Engineer specializing in deep learning and signal processing for EEG analysis. Lead Maintainer of the widely used Braindecode library and Core Developer for the Mother of all BCI Benchamark - MOABB benchmarking framework, actively shaping standards and enabling reproducible research in brain-computer interfaces and EEG decoding. Experience in deep learning and machine learning applied to EEG decoding, validated through peer-reviewed publications, community engagement with reviewer at NeurIPS, ICLR, ICML and more journals.
- Core developer of Braindecode and Mother of BCI Benchmark Python toolkits. Go opensource!


## 2. Dung Truong

- Affiliation: Swartz Center for Computational Neuroscience, University of California, San Diego (UCSD)
- E-mail: dutruong@ucsd.edu
- Research Engineer at Swartz Center for Computational Neuroscience.
- Specialize in the standardization and large-scale processing of EEG data, with a research focus on developing deep learning algorithms for EEG decoding-particularly in the areas of representation learning using self-supervised methods and generative models.
- Core developer and maintainer of multiple open-source and neuroinformatic projects, including HED, EEGLAB, EEGDash and NEMAR.


## 3. Pierre Guetschel

- Affiliation: Donders Institute for Brain, Cognition and Behaviour, Radboud University, Netherlands
- E-mail: pierre.guetschel@donders.ru.nl
- Pierre Guetschel is a PhD candidate at the Donders Institute. His research interests are on developing deep learning algorithms for EEG decoding, with a special focus on transfer learning, self-supervised learning and foundation models.
- Core developer of Braindecode and Mother of BCI Benchmark Python toolkits.


## 4. Seyed Yahya Shirazi

- Affiliation: Swartz Center for Computational Neuroscience, Institute for Neural Computation, University of California San Diego, CA, USA
- E-mail: shirazi@ieee.org
- Yahya is an Assistant Project Scientist at UC San Diego, specializing in Brain-Behavior research, computational neuroscience, and neuroinformatics initiatives.
- Led the HBN-EEG data curation and annotation.
- Lead Scientist for BIDS extension proposals to EMG and Stimulus. Core member of the HED working group. Core member of the EEGLAB development team


## 5. Isabelle Guyon

- Affiliation: Université Paris-Saclay, France. ChaLearn, California. Google DeepMind, California.

- E-mail: guyon@chalearn.org
- Isabelle Guyon is Director, Research Scientist at Google DeepMind, in detachment from her position as professor of Artificial Intelligence at Université Paris-Saclay (Orsay). She specializes in data-centric AI, statistical data analysis, pattern recognition, and machine learning. Her areas of expertise include computer vision, bioinformatics, and power systems. She has been a strong promoter of challenges and benchmarks. Her recent interests include AIassisted human communication. Prior to joining Paris-Saclay she worked as an independent consultant and was a researcher at AT\&T Bell Laboratories, where she pioneered applications of neural networks in the 80's with Yann LeCun and Yoshua Bengio, among others. She is president of Chalearn, a non-profit dedicated to organizing challenges in machine learning, community lead of Codalab competitions a challenge organization platform, action editor of the Journal of Machine Learning Research (JMLR), co-editor of the Data-Centric Machine Learning Research Journal (DMLR), and served as program co-chair of NIPS 2016 and general co-chair of NIPS 2017. She is the 2020 recipient of the BBVA Frontiers in Research Award together with Prof. Schoelkopf and Prof. Vapnik for contributions to kernel methods (including the invention of Support Vector Machines - SVM) and to causality in machine learning.
- Expert Advisor.


# 6. Alan Evans 

- Affiliation: McGill University, Canada
- E-mail: alan.evans@mcgill.ca
- Distinguished James McGill Professor of Neurology and Psychiatry at McGill U.
- Director, EEGNet consortium; expert in structural brain network modeling.
- Co-director, Global Brain Consortium (GBC).
- Director, Canadian Open Neuroscience Platform


## 7. Pedro Antonio Valdés-Sosa

- Affiliation: Cuban Neuroscience Center (CNEURO) and the University of Electronic Science and Technology of China (UESTC)
- E-mail: pedro.valdes@neuroinformatics-collaboratory.org
- Pioneer in EEG source localization, Bayesian modeling, and brain connectivity analysis.
- Co-director of the Global Brain Consortium (GBC) and contributor to EEG-based BCI research.


## 8. Scott Makeig

- Affiliation: Swartz Center for Computational Neuroscience, University of California, San Diego (UCSD)
- E-mail: smakeig@ucsd.edu
- Pioneer in EEG analysis and development of Independent Component Analysis (ICA) for brain signal decomposition.
- Leader in studying brain dynamics and mobile brain/body imaging (MoBI).


## 9. Alexandre Gramfort

- Affiliation: Meta Reality Labs, Paris, France
- E-mail: agramfort@meta.com
- Senior Research Scientist, Meta, (Ex-) Research director, HdR, Inria, MIND Team, Univ. Paris Saclay
- Currently senior research scientist manager at Meta Reality Labs in Paris. Works on machine learning technologies to decode surface EMG signals. Previously was research director (DR, HdR) at Inria, leading the MIND Team, known formerly as Parietal. Works on statistical machine learning, signal processing, optimization, scientific computing and software engineering with primary applications in neuroscience and biosignal processing (in particular EEG).


## 10. Jean-Rémi King

- Affiliation: Brain \& AI team, FAIR, Meta, Paris, France
- E-mail: jeanremi@meta.com
- Research Scientist, Meta

- CNRS researcher at École Normale Supérieure currently detached to Meta AI, where he leads the Brain \& AI team. This team aims to identify the brain and computational bases of human intelligence, with a focus on language. For this, they develop deep learning algorithms to decode and model brain activity recorded with MEG, EEG, electrophysiology and fMRI.


# 11. Terrence J Sejnowski 

- Affiliation: Salk Institute for Biological Studies and the University of California, San Diego (UCSD)
- E-mail: terry@salk.edu
- Co-developed the Boltzmann machine and contributed foundational work to deep learning.
- Instrumental in bridging neuroscience and machine learning.


## 12. Marie-Constance Corsi

- Affiliation: NERV team, Sorbonne Université, Institut du Cerveau - Paris Brain Institute ICM, CNRS, Inria, Inserm, AP-HP, Hôpital de la Pitié-Salpêtrière, Paris, France
- Email: marie-constance.corsi@inria.fr
- Inria research scientist at Paris Brain Institute in the NERV Lab. Her research aims to enhance Brain-Computer Interfaces (BCIs) by identifying neurophysiological markers of training and integrating multimodal data for better classifier information. Additionally, she is developing interpretable AI tools for diagnosing neurological diseases.


## 13. Michael P. Milham

- Affiliation:
- Email: michael.milham@nki.rfmh.org
- Michael P. Milham, MD, PhD, is an internationally recognized neuroscience researcher, a child and adolescent psychiatrist, and Director for the Center of Biomedical Imaging and Neuromodulation at the Nathan S. Kline Institute for Psychiatric Research. Dr. Milham is one of our nation's most prolific neuroimaging researchers, with over 200 articles published since 2005 and recognition as a Clarivate Highly Cited Researcher (top $.1 \%$ for Neuroscience and Behavior) every year since 2014. He was also the recipient of the Organization of Human Brain Mapping's highly prestigious Wiley Young Investigator Award in 2014.


## 14. Alexandre R Franco

- Affiliation: Data Informatics and Sharing of Knowledge Core, Child Mind Institute, USA
- Affiliation: Computational Neuroimaging Laboratories, Nathan Kline Institute for Psychiatric Research, USA
- Email: alexandre.franco@childmind.org
- Alexandre R. Franco, PhD is the Director of the Data Informatics and Sharing of Knowledge Core at the Child Mind Institute and Director of the Computational Neuroimaging Laboratories at the Nathan Kline Institute. His research focuses on optimizing MRI and EEG data collection, data processing and data sharing.
- Leads the efforts for the International Neuroimaging Data-sharing Initiative.


## 15. Aviv Dotan

- Affiliation: Dept. of Cognitive and Brain Sciences, Ben-Gurion University of the Negev, Beer-Sheva, Israel
- Email: avivdot@post.bgu.ac.il
- Postdoctoral researcher in the Computational Psychiatry and Neurotechnology Lab at BGU.
- Expert in machine learning and deep learning applications to EEG data.


## 16. Oren Shriki

- Affiliation: Dept. of Cognitive and Brain Sciences, Ben-Gurion University of the Negev, Beer-Sheva, Israel
- Email: shrikio@bgu.ac.il
- Head of the Computational Psychiatry and Neurotechnology Lab at BGU.
- Leader in the development and application of neurotechnology.


## 17. Amit Majumdar

- Affiliation: San Diego Supercomputer Center (SDSC), University of California, San Diego (UCSD)
- Email: majumdar@sdsc.edu
- Director of SDSC's Data Enabled Scientific Computing (DESC) division.
- Director of the Neuroscience Gateway (NSG).


# 18. Sylvain Chevallier 

- Affiliation: A\&O/TAU team director LISN, Université Paris-Saclay, Inria TAU, France
- Email: sylvain.a.chevallier@inria.fr
- Full professor at the University Paris-Saclay, a board member of the DATAIA/ClusterIA Institute of Saclay, and a co-leader of the TAU team. He works on frugal learning and transfer learning, which are applied to time series analysis with applications such as BCI computing. He contributed to several open-source Python toolboxes for signal processing and machine learning using Riemannian geometry. He is leading the Codalab/Codabench framework for benchmark and data competition.


## 19. Arnaud Delorme

- Affiliation: University of California, San Diego (UCSD); Paul Sabatier University, France
- Email: arno@ucsd.edu
- Leads the EEGLAB project, a widely adopted open-source toolbox for EEG data analysis.
- Research focuses on advanced EEG methodologies, integrating AI, and exploring the neuroscience of meditation and mind wandering.


## References

[1] Thomas M Achenbach. The child behavior checklist and related instruments. The use of psychological testing for treatment planning and outcomes assessment, 1999.
[2] Hamdi Altaheri, Ghulam Muhammad, and Mansour Alsulaiman. Physics-informed attention temporal convolutional network for EEG-based motor imagery classification. IEEE transactions on industrial informatics, 19(2):2249-2258, 2022.
[3] Andrea Apicella, Pasquale Arpaia, Giovanni D’Errico, Davide Marocco, Giovanna Mastrati, Nicola Moccaldi, and Roberto Prevete. Toward cross-subject and cross-session generalization in EEG-based emotion recognition: Systematic review, taxonomy, and methods. arXiv [cs.LG], 16 December 2022.
[4] Bruno Aristimunha, Igor Carrara, Pierre Guetschel, Sara Sedlar, Pedro Rodrigues, Jan Sosulski, Divyesh Narayanan, Erik Bjareholt, Barthelemy Quentin, Robin Tibor Schirrmeister, Emmanuel Kalunga, Ludovic Darmet, Cattan Gregoire, Ali Abdul Hussain, Ramiro Gatti, Vladislav Goncharenko, Jordy Thielen, Thomas Moreau, Yannick Roy, Vinay Jayaram, Alexandre Barachant, and Sylvain Chevallier. Mother of all BCI Benchmarks, 2023. URL https://github.com/NeuroTechX/moabb.
[5] Bruno Aristimunha, Raphael Y de Camargo, Walter H Lopez Pinaya, Sylvain Chevallier, Alexandre Gramfort, and Cedric Rommel. Evaluating the structure of cognitive tasks with transfer learning. arXiv preprint arXiv:2308.02408, 2023.
[6] Bruno Aristimunha, Thomas Moreau, Sylvain Chevallier, Raphael Y de Camargo, and MarieConstance Corsi. What is the best model for decoding neurophysiological signals? Depends on how you evaluate. In 33rd Annual Computational Neuroscience Meeting* CNS, 2024.
[7] Hubert Banville, Yohann Benchetrit, Stéphane d'Ascoli, Jérémy Rapin, and Jean-Rémi King. Scaling laws for decoding images from brain activity. arXiv preprint arXiv:2501.15322, 2025.
[8] Philipp Bomatter and Henry Gouk. Is limited participant diversity impeding eeg-based machine learning? arXiv preprint arXiv:2503.13497, 2025.
[9] Philipp Bomatter, Joseph Paillard, Pilar Garces, Jörg Hipp, and Denis-Alexander Engemann. Machine learning of brain-specific biomarkers from EEG. EBioMedicine, 106, 2024.

[10] Hosted by WiDS Worldwide: https://www.widsworldwide.org/. WiDS Datathon 2025, Unraveling the Mysteries of the Female Brain: Sex Patterns in ADHD. https://kaggle.com/ competitions/widsdatathon2025, 2025. Kaggle.
[11] David E Carlson, Ricardo Chavarriaga, Yiling Liu, Fabien Lotte, and Bao-Liang Lu. The nerveml (neural engineering reproducibility and validity essentials for machine learning) checklist: ensuring machine learning advances neural engineering*. Journal of Neural Engineering, 22(2): 021002, mar 2025. doi: 10.1088/1741-2552/adbfbd. URL https://dx.doi.org/10.1088/ 1741-2552/adbfbd.
[12] Avshalom Caspi, Renate M Houts, Daniel W Belsky, Sidra J Goldman-Mellor, Honalee Harrington, Salomon Israel, Madeline H Meier, Sandhya Ramrakha, Idan Shalev, Richie Poulton, and Terrie E Moffitt. The p factor: One general psychopathology factor in the structure of psychiatric disorders?: One general psychopathology factor in the structure of psychiatric disorders? Clin. Psychol. Sci., 2(2):119-137, March 2014. doi: 10.1177/2167702613497473.
[13] Xia Chen, Xiangbin Teng, Han Chen, Yafeng Pan, and Philipp Geyer. Toward reliable signals decoding for electroencephalogram: A benchmark study to eegnex. Biomedical Signal Processing and Control, 87:105475, 2024. ISSN 1746-8094. doi: https://doi.org/10.1016/ j.bspc.2023.105475. URL https://www.sciencedirect.com/science/article/pii/ S1746809423009084.
[14] Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar, Bruna Lopes, Sébastien Velut, Salim Khazem, and Thomas Moreau. The largest eeg-based bci reproducibility study for open science: the moabb benchmark. arXiv preprint arXiv:2404.15319, 2024.
[15] Bruce N Cuthbert. Research domain criteria: toward future psychiatric nosologies. Dialogues Clin. Neurosci., 17(1):89-97, March 2015. doi: 10.31887/dcns.2015.17.1/bcuthbert.
[16] Stéphane d'Ascoli, Corentin Bel, Jérémy Rapin, Hubert Banville, Yohann Benchetrit, Christophe Pallier, and Jean-Rémi King. Decoding individual words from non-invasive brain recordings across 723 participants. arXiv preprint arXiv:2412.17829, 2024.
[17] Arnaud Delorme. EEG is better left alone. Scientific reports, 13(1):2372, 2023.
[18] Arnaud Delorme, Dung Truong, Choonhan Youn, Subha Sivagnanam, Kenneth Yoshimoto, Russell A Poldrack, Amit Majumdar, and Scott Makeig. NEMAR: An open access data, tools, and compute resource operating on NeuroElectroMagnetic data. arXiv [q-bio.QM], 4 March 2022.
[19] Théophile Demazure, Alexander J Karran, Jared Boasen, Pierre-Majorique Léger, and Sylvain Sénécal. Distributed remote EEG data collection for NeuroIS research: A methodological framework. In Augmented Cognition, Lecture notes in computer science, pages 3-22. Springer International Publishing, Cham, 2021. doi: 10.1007/978-3-030-78114-91_1.
[20] Yi Ding, Nigel Wei Jun Ang, Aung Aung Phyo Wai, and Cuntai Guan. Learning generalized representations of eeg between multiple cognitive attention tasks. In 2021 43rd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC), pages 306-310. IEEE, 2021.
[21] Charlotte DiStefano, Abigail Dickinson, Elizabeth Baker, and Shafali Spurling Jeste. EEG data collection in children with ASD: The role of state in data quality and spectral power. Res. Autism Spectr. Disord., 57:132-144, 1 January 2019. doi: 10.1016/j.rasd.2018.10.001.
[22] Lukas Gemein, Sinead Gaubert, Claire Paquet, Joseph Paillard, Sebastian C Holst, Thomas Tveitstol, Ira Haraldsen, David Hawellek, Joerg Hipp, and Denis Engemann. Neurologically altered brain activity may not look like aged brain activity: Implications for brain-age modeling and biomarker strategies. bioRxiv, 2025. doi: 10.1101/2025.04.15.648903. URL https : //www.biorxiv.org/content/early/2025/04/20/2025.04.15.648903.
[23] Lukas AW Gemein, Robin T Schirrmeister, Patryk Chrabąszcz, Daniel Wilson, Joschka Boedecker, Andreas Schulze-Bonhage, Frank Hutter, and Tonio Ball. Machine-learning-based diagnostics of EEG pathology. NeuroImage, 220:117021, 2020.

[24] Krzysztof J Gorgolewski, Tibor Auer, Vince D Calhoun, R Cameron Craddock, Samir Das, Eugene P Duff, Guillaume Flandin, Satrajit S Ghosh, Tristan Glatard, Yaroslav O Halchenko, Daniel A Handwerker, Michael Hanke, David Keator, Xiangrui Li, Zachary Michael, Camille Maumet, B Nolan Nichols, Thomas E Nichols, John Pellman, Jean-Baptiste Poline, Ariel Rokem, Gunnar Schaefer, Vanessa Sochat, William Triplett, Jessica A Turner, Gaël Varoquaux, and Russell A Poldrack. The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Sci Data, 3:160044, 21 June 2016. doi: 10.1038/sdata. 2016.44.
[25] Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A Engemann, Daniel Strohmeier, Christian Brodbeck, Lauri Parkkonen, and Matti S Hämäläinen. MNE software for processing MEG and EEG data. NeuroImage, 86:446-460, 2014.
[26] NTX Hackathon. Sleep states. https://www.codabench.org/competitions/1777, 2023. Accessed: 2025-04-08.
[27] Jinwoo Hong, Jundong Hwang, and Jong-Hwan Lee. General psychopathology factor (pfactor) prediction using resting-state functional connectivity and a scanner-generalization neural network. Journal of Psychiatric Research, 158:114-125, 2023.
[28] Thomas Insel, Bruce Cuthbert, Marjorie Garvey, Robert Heinssen, Daniel S Pine, Kevin Quinn, Charles Sanislow, and Philip Wang. Research domain criteria (RDoC): toward a new classification framework for research on mental disorders. Am. J. Psychiatry, 167(7):748-751, July 2010. doi: 10.1176/appi.ajp.2010.09091379.
[29] Weibang Jiang, Liming Zhao, and Bao liang Lu. Large brain model for learning generic representations with tremendous EEG data in BCI. In The Twelfth International Conference on Learning Representations, 2024.
[30] Stephen B Johnson, Glen Whitney, Matthew McAuliffe, Hailong Wang, Evan McCreedy, Leon Rozenblit, and Clark C Evans. Using global unique identifiers to link autism collections. J. Am. Med. Inform. Assoc., 17(6):689-695, 1 November 2010. doi: 10.1136/jamia.2009.002063.
[31] Roman Kessler, Alexander Enge, and Michael A Skeide. How EEG preprocessing shapes decoding performance. arXiv preprint arXiv:2410.14453, 2024.
[32] Ann-Kathrin Kiessner, Robin T Schirrmeister, Joschka Boedecker, and Tonio Ball. Reaching the ceiling? Empirical scaling behaviour for deep EEG pathology classification. Computers in Biology and Medicine, 178:108681, 2024.
[33] Vernon J Lawhern, Amelia J Solon, Nicholas R Waytowich, Stephen M Gordon, Chou P Hung, and Brent J Lance. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of neural engineering, 15(5):056013, 2018.
[34] Sandra K Loo, Agatha Lenartowicz, and Scott Makeig. Research review: Use of EEG biomarkers in child psychiatry research-current state and future directions. Journal of Child Psychology and Psychiatry, 57(1):4-17, 2016.
[35] Scott Makeig and Kay Robbins. Events in context—The HED framework for the study of brain, experience and behavior. Frontiers in Neuroinformatics, 18:1292667, 2024.
[36] Kaare B Mikkelsen, Yousef R Tabar, Christian B Christensen, and Preben Kidmose. EEGs vary less between lab and home locations than they do between people. Front. Comput. Neurosci., 15:565244, 16 February 2021. doi: 10.3389/fncom.2021.565244.
[37] INRIA NeuroTechX and TAILOR network. Brain age prediction challenge from eeg. https : //codalab.lisn.upsaclay.fr/competitions/8336, 2022. Accessed: 2025-04-08.
[38] R C Oldfield. The assessment and analysis of handedness: the edinburgh inventory. Neuropsychologia, 9(1):97-113, March 1971. doi: 10.1016/0028-3932(71)90067-4.

[39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.
[40] Cyril R Pernet, Stefan Appelhoff, Krzysztof J Gorgolewski, Guillaume Flandin, Christophe Phillips, Arnaud Delorme, and Robert Oostenveld. EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6(1):103, 25 June 2019. doi: $10.1038 / \mathrm{s} 41597-019-0104-8$.
[41] Colin Reilly, Patricia Atkinson, Ayesha Memon, Chloe Jones, Lyvia Dabydeen, J Helen Cross, Krishna B Das, Christopher Gillberg, Brian G R Neville, and Rod C Scott. Autism, ADHD and parent-reported behavioural difficulties in young children with epilepsy. Seizure, 71:233-239, 1 October 2019. doi: 10.1016/j.seizure.2019.08.003.
[42] Kay Robbins, Dung Truong, Stefan Appelhoff, Arnaud Delorme, and Scott Makeig. Capturing the nature of events and event context using hierarchical event descriptors (HED). Neuroimage, 245:118766, 15 December 2021. doi: 10.1016/j.neuroimage.2021.118766.
[43] David Sabbagh, Pierre Ablin, Gaël Varoquaux, Alexandre Gramfort, and Denis A Engemann. Predictive regression modeling with MEG/EEG: from source power to signals and cognitive states. NeuroImage, 222:116893, 2020.
[44] Simanto Saha and Mathias Baumert. Intra- and inter-subject variability in EEG-based sensorimotor brain computer interface: A review. Front. Comput. Neurosci., 13:87, 2019. doi: 10.3389/fncom.2019.00087.
[45] Eduardo Santamaria-Vazquez, Victor Martinez-Cagigal, Fernando Vaquerizo-Villar, and Roberto Hornero. EEG-inception: a novel deep convolutional neural network for assistive ERPbased brain-computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 28(12):2773-2782, 2020.
[46] Adam Santorelli, Arianna Zuanazzi, Michael Leyden, Logan Lawler, Maggie Devkin, Yuki Kotani, and Gregory Kiar. Child Mind Institute - Problematic Internet Use. https://kaggle. com/competitions/child-mind-institute-problematic-internet-use, 2024. Kaggle.
[47] Robin Tibor Schirrmeister, Jost Tobias Springenberg, Lukas Dominique Josef Fiederer, Martin Glasstetter, Katharina Eggensperger, Michael Tangermann, Frank Hutter, Wolfram Burgard, and Tonio Ball. Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping, 38(11):5391-5420, 2017.
[48] Maurício Scopel Hoffmann, Tyler Maxwell Moore, Luiza Kvitko Axelrud, Nim Tottenham, Xi-Nian Zuo, Luis Augusto Rohde, Michael Peter Milham, Theodore Daniel Satterthwaite, and Giovanni Abrahão Salum. Reliability and validity of bifactor models of dimensional psychopathology in youth. J. Psychopathol. Clin. Sci., 131(4):407-421, May 2022. doi: 10.1037/abn0000749.
[49] Seyed Yahya Shirazi, Alexandre Franco, Mauricio Scopel Hoffmann, Nathalia Esper, Dung Truong, Arnaud Delorme, Michael Milham, and Scott Makeig. HBN-EEG: The FAIR implementation of the healthy brain network (HBN) electroencephalography dataset. bioRxiv, page 2024.10.03.615261, 3 October 2024. doi: 10.1101/2024.10.03.615261.
[50] Yonghao Song, Qingqing Zheng, Bingchuan Liu, and Xiaorong Gao. EEG conformer: Convolutional transformer for EEG decoding and visualization. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31:710-719, 2022.
[51] Darinka Trübutschek, Yu-Fang Yang, Claudia Gianelli, Elena Cesnaite, Nastassja L Fischer, Mikkel C Vinding, Tom R Marshall, Johannes Algermissen, Annalisa Pascarella, Tuomas

Puoliväli, et al. EEGManyPipelines: a large-scale, grassroots multi-analyst study of electroencephalography analysis practices in the wild. Journal of Cognitive Neuroscience, 36(2): $217-224,2024$.
[52] Özgün Turgut, Felix S Bott, Markus Ploner, and Daniel Rueckert. Are foundation models useful feature extractors for electroencephalography analysis? arXiv preprint arXiv:2502.21086, 2025.
[53] Yihe Wang, Nan Huang, Nadia Mammone, Marco Cecchi, and Xiang Zhang. LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection. arXiv preprint arXiv:2502.01678, 2025.
[54] Xiaoxi Wei, A. Aldo Faisal, Moritz Grosse-Wentrup, Alexandre Gramfort, Sylvain Chevallier, Vinay Jayaram, Camille Jeunet, Stylianos Bakas, Siegfried Ludwig, Konstantinos Barmpas, Mehdi Bahri, Yannis Panagakis, Nikolaos Laskaris, Dimitrios A. Adamos, Stefanos Zafeiriou, William C. Duong, Stephen M. Gordon, Vernon J. Lawhern, Maciej Śliwowski, Vincent Rouanne, and Piotr Tempczyk. 2021 BEETL Competition: Advancing Transfer Learning for Subject Independence and Heterogenous EEG Data Sets. In Douwe Kiela, Marco Ciccone, and Barbara Caputo, editors, Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track, volume 176 of Proceedings of Machine Learning Research, pages 205-219. PMLR, 06-14 Dec 2022. URL https://proceedings.mlr.press/v176/wei22a.html.
[55] Chaoqi Yang, M Westover, and Jimeng Sun. Biot: Biosignal transformer for cross-data learning in the wild. Advances in Neural Information Processing Systems, 36:78240-78260, 2023.
[56] Hilmar G Zech, Markus Reichert, Ulrich W Ebner-Priemer, Heike Tost, Michael A Rapp, Andreas Heinz, Raymond J Dolan, Michael N Smolka, and Lorenz Deserno. Mobile data collection of cognitive-behavioral tasks in substance use disorders: Where are we now? Neuropsychobiology, 81(5):438-450, 29 March 2022. doi: 10.1159/000523697.
[57] Xinliang Zhou, Chenyu Liu, Zhisheng Chen, Kun Wang, Yi Ding, Ziyu Jia, and Qingsong Wen. Brain Foundation Models: A Survey on Advancements in Neural Signal Processing and Brain Discovery. arXiv preprint arXiv:2503.00580, 2025.

---

*Note: This document contains 1 extracted image(s). In the ZIP download, images are organized in the `images/` folder for proper markdown reference.*

---

