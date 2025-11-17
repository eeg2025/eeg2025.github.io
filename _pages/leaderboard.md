---
layout: page
permalink: /leaderboard/
title: Leaderboard
nav: true
nav_order: 9
---

## Final Leaderboard

Thank you so much for participating in the 2025 NeurIPS EEG competition. There were 1,183 teams/participants and more than 8,000 submissions on the open source platform Codabench. CodaBench is the newer version of “Codalab competitions” (ranked first by mlcontests.com, ahead of Kaggle and Tianchi of Alibaba), according to mlcontests.com. Our competition is by far the largest, having been organized on Codabench, so far. Our EEGDash library was installed more than 67,000 times, facilitating open-source data transfer and restructuring for AI/ML training and inference. By every metric, this competition was a tremendous success, and we thank you for your contribution.

The jury reviewed the top entries, and no changes were made to the rankings of the leading teams based on that analysis. To maintain transparency and clarify prize distribution, we have updated the final prize structure.

As organizers, we made an error by not randomizing samples in Challenge 2, which allowed some teams to exploit the fact that contiguous trials likely came from the same subjects. During office hours and on the forum, several contestants inquired whether using this information was permitted; however, due to a communication breakdown, they received contradictory answers. Even with this advantage, Challenge 2 remained extremely difficult – only three teams achieved scores below 0.99, our threshold for including this challenge in the final scoring (recall that a score of 1 represents predicting the mean target value).
Therefore, we have decided to award Challenge 1 and Challenge 2 separately. To accommodate this new structure, we secured an additional $2,500 cash prize for the prize pool. Cash prizes and travel awards will go to the top two teams in Challenge 1 and the top two teams in Challenge 2. Thanks to Meta for sponsoring these awards.

The winners are as follows:

**Challenge 1:**
1. 🏆 Team KUL_EEG: 0.88668
2. 🥈 Team Sigma Nova: 0.90932
3. 🥉 Team MIN~C² (MIND-CICO): 0.91026
4. Team Meta Brain & AI: 0.9106
5. Team MBZUAI \[dsml.kz\]: 0.91394
6. Team BCILab: 0.91856
7. Team MostlyBerlin: 0.91946
8. Team Team Marque: 0.91982
9. Team CyberBobBeta: 0.92112
10. Team JLShen: 0.92991

**Challenge 2:**
1. 🥇 Team JLShen: 0.97843
2. 🥈 Team MBZUAI [dsml.kz] : 0.98519
3. 🥉 Team MIN~C² (MIND-CICO): 0.98817

The **diversity prize** is awarded to team RIML, with a score on Challenge 1 of 0.93106. This award, even if it comes with no monetary reward, is a recognition of the team's hard work and an encouragement to take on more challenges.

Again, thank you for putting so much energy into this challenge. Keep your models ready, as we are planning other challenges.

The 2025 EEG Organizing Committee


## Interactive Leaderboard

{% comment %}
This page includes a pre-generated HTML snippet with a Plotly chart. The
snippet is created by build_leaderboard_html.py after fetching Codabench
results. If the include is missing, run the scripts locally or wait for CI.
{% endcomment %}

Interactive leaderboard summary from Codabench results.

See all [results](https://www.codabench.org/competitions/9975/#/results-tab) on CodaBench.

{% include leaderboard_generated.html %}
