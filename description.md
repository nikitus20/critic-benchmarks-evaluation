# Project Kick-off: Implementation Strategy for Phase 1 Analysis

## Objective for This Stage

The primary goal of this initial phase is to empirically validate our core thesis: that current mathematical reasoning benchmarks, specifically DeltaBench and ProcessBench, contain a significant "early error bias" which fundamentally flaws their ability to measure a model's true, sustained reasoning capabilities. We will demonstrate this not only by analyzing the data distribution but also by showing how model performance, under the benchmarks' own metrics, degrades as a function of error depth.

## 1. Analysis of Error Distribution in DeltaBench

**Objective:** To quantify the distribution of error locations within the long Chain-of-Thought (CoT) solutions in the DeltaBench dataset. This will be the first piece of evidence to assess whether a positional bias exists.

### Methodology

**Data Acquisition:** We will programmatically access the DeltaBench dataset (1,236 samples) from its official Hugging Face repository. 

**Data Parsing:** For each sample containing an error, we will parse the human annotations. The approach should follow original implementation technique. 

**Positional Calculation:** We compute 2 measures of error depth -- section level and step level. For the second we need to track total number of steps as well as error step. If error step is not explicitly provided in data, we should approximate it with the middle step in its corresponding section.

**Required Resources:** For this part we need data access. For later critic performance valuation we need API access to models. API keys are stored in .env file. 

**Expected Outcome & Visualization:** Histograms plotting the absolute/relative error position on section and step level against the frequency of errors. This will give us a clear visual representation of where errors are concentrated in DeltaBench solutions. The plots should be split by problem topic, i.e. math/code/pcb/general
## 2. Analysis of Error Distribution in ProcessBench

**Objective:** To replicate and programmatically confirm the early error bias already visualized in the ProcessBench paper. This step validates our understanding of their data and provides a baseline for our own analysis.

### Methodology

**Data Acquisition:** We will access the ProcessBench dataset (3,400 samples) via its Hugging Face repository.

**Data Parsing:** The ProcessBench format is more direct. Each sample contains a list of steps and a label field, where the label is the 0-indexed step number of the earliest error (or -1 if correct).

**Positional Calculation:** We will calculate the relative error position for each erroneous sample.

**Required Resources:** Access to the ProcessBench dataset and its evaluation code.

**Expected Outcome & Visualization:** A histogram confirming the findings from the original paper, which shows a heavy concentration of errors in the first few steps (typically steps 0-5). This will serve as the strongest initial evidence for our thesis.

## 3. Analysis of DeltaBench Metric Dependence on Error Depth

**Objective:** To measure how critic performance, according to the official DeltaBench F1 score, changes as errors appear later in the reasoning process.

### Methodology

**Stratify Data:** Using the relative error positions calculated in step 1, we will partition the DeltaBench dataset into three buckets:
- Early Errors (0-33%)
- Middle Errors (33-67%)
- Late Errors (67-100%)

**Model Evaluation:** We will run a the best critic model (e.g., GPT-4o-mini) against each of these stratified buckets.

**Metric Calculation:** Using the official DeltaBench evaluation scripts, we will calculate the F1 score (based on precision and recall of detecting an erroneous section) for each model on each of the three buckets separately. 

**Required Resources:** Official DeltaBench evaluation code, API access to critic models.

**Expected Outcome & Visualization:** A bar chart comparing the F1 scores across the "Early," "Middle," and "Late" error buckets. Our hypothesis is that we will observe a significant performance drop for all models on the "Late Errors" bucket. Moreover, the data should be split into math/code/pcb/general parts, as they show different performance. 

## 4. Analysis of ProcessBench Metric Dependence on Error Depth

**Objective:** To measure how critic performance, according to the official ProcessBench "exact identification" metric, degrades with error depth.

### Methodology

**Stratify Data:** We will partition the ProcessBench dataset into the same "Early," "Middle," and "Late" error buckets based on the relative error positions calculated in step 2.

**Model Evaluation:** We will evaluate the same set of critic models on these stratified ProcessBench buckets.

**Metric Calculation:** Using the ProcessBench evaluation framework, we will calculate the exact step identification accuracy for each model on each bucket. This metric measures whether the model correctly identifies the precise index of the first error.

**Required Resources:** Official ProcessBench evaluation code, API access to critic models.

**Expected Outcome & Visualization:** A bar chart comparing the exact identification accuracy for each model across the three error buckets. We predict a sharp decline in accuracy for the "Late Errors" bucket, which would powerfully demonstrate that current models are not skilled at sustained, long-range error detection.

## Additional Aims & Sanity Checks

To further strengthen our initial findings and preemptively address potential questions, we should also aim to complete the following:

### The "Idiotic Critic" Sanity Check

We will test a simple, non-reasoning heuristic critic on DeltaBench. Mainly, it should be a critic that responds with sections [3, ..., 36]. 

Moreover, we will try to enhance ProcessBench critic performance by signifying early error bias. For example, we should try to specifically prompt the model to double check initial steps really hard, and perfect the prompt engineering to achieve best results.

### Preliminary Error Type Analysis

We will sample 50 "early" and 50 "late" errors from both benchmarks and use a high-capability model (like GPT-4o) to perform a preliminary classification into Computational Errors vs. Logical/Reasoning Errors. This aligns with taxonomies used in other benchmarks like ErrorRadar. This will provide early, qualitative evidence that late-stage errors are not only rarer but also conceptually different and likely more difficult to detect.

