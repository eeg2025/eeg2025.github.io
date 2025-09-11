---
layout: page
permalink: /leaderboard/
title: Leaderboard
description: Interactive leaderboard summary from Codabench results.
nav: true
nav_order: 7
---

{% comment %}
This page includes a pre-generated HTML snippet with a Plotly chart. The
snippet is created by build_leaderboard_html.py after fetching Codabench
results. If the include is missing, run the scripts locally or wait for CI.
{% endcomment %}

See all [results](https://www.codabench.org/competitions/9975/#/results-tab) on CodaBench.

{% include leaderboard_generated.html %}
