# Evaluating LLM Code Generation for Scientific Computing in Julia

This repository is the replication package for our paper evaluating how well
large language models generate Julia code for scientific-computing problems.

**What it contains**
- 14 mathematically grounded tasks adapted from the SciCode benchmark (originally Python) for Julia
- 6 LLMs accessed through the Groq API, with 5 independently generated samples per model and task (420 observations in total)
- The experiment script that generates code, runs it in Julia and scores the results
- The raw results (experiment_results.csv) and the statistical analysis (ANOVA with Tukey HSD, plus Kruskal-Wallis as a robustness check)

**Metrics**
Correctness rate, numerical accuracy (relative L2 error), execution time, consistency across samples, pass@k, and an n-gram match score against reference implementations (available for 3 of the 14 tasks).

**Models evaluated**
Llama 3.3 70B, Llama 4 Scout, Qwen3 32B, DeepSeek R1, Mixtral 8x7B and Llama 3.1 8B.
Note: two models originally planned for the study were decommissioned by the
provider during data collection; the models listed here are the ones that
actually produced the results.

**Requirements and how to run**
See the sections below for the Julia version, Python packages and the order of steps.
You need your own Groq API key, set as the environment variable GROQ_API_KEY.
