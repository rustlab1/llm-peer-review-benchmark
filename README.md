# LLM peer review benchmark

Code and prompts for **Large language models detect methodological errors that human peer review misses in clinical trials**.

Dominikus Brian and Xavier Isidro Pereira Hernández contributed equally. The other authors are Hao Yin, Jeffrey Koury and Ruslan Rust. The preprint is being prepared for submission.

The benchmark tests 15 models on three historical trial manuscripts and 58 randomized trials published in 2026, including all eligible phase 3 trials with open-access full text published between July and October 2026. All 15 models reviewed the historical manuscripts and a subset of three 2026 trials four times each, with unmodified controls (405 reviews). Twelve models then reviewed 55 further trials in their error-seeded and unmodified versions, and reviewed the subset of three again through the same pipeline (1,391 reviews). In total, 1,796 reviews were scored by two blinded LLM judges. This repository contains the code and prompts for the current manuscript. The study data, including the further trials (folder `further_trials/`), are in Supplementary Data 1.

## Reproduce the results

Use Python 3.10 or later and the score file from Supplementary Data 1:

```bash
python -m pip install -r requirements.txt
python reproduce.py --scores /path/to/Supplementary_Data_1/validation/all_judge_item_scores.csv
```

This writes per-model results, judge agreement and correlations to `results/`, using saved scores without calling model APIs.

The primary analysis averages the scores of Fable 5 and Opus 5. Binary detection requires their mean to be at least 0.5. The historical human comparator uses the untrained control group. Flags on control manuscripts may be valid criticisms, and judge agreement does not establish accuracy against human labels.

## Reproduce the 58-trial results

These scripts use the saved reviews and judge outputs in `further_trials/` of Supplementary Data 1, without calling model APIs:

```bash
python code/further_trials/validate_further_trials.py /path/to/Supplementary_Data_1/further_trials
python code/further_trials/prepare_workspace.py /path/to/Supplementary_Data_1 work_further_trials
export SD1=/path/to/Supplementary_Data_1 FT_WORK=work_further_trials
bash code/further_trials/run_final_analysis.sh
```

The validator checks that every trial has its reviews and judge outputs and recalculates each model's results on the 55 further trials. `run_final_analysis.sh` writes the results behind Figure 2, Supplementary Figures 4 to 6 and 8 and the numbers in the text to `work_further_trials/data/`, and the figures to `work_further_trials/figures/`. With the default 10,000 bootstrap resamples the run takes a few hours; `NBOOT=200` reproduces every point estimate in about five minutes, and only the confidence intervals change.

## Generate the figures

Copy `data/`, `data_original/` and `additional_judges/` from Supplementary Data 1 into the repository root (or set `BMJ_ROOT=/path/to/Supplementary_Data_1`), then run:

```bash
python code/make_figures.py
python code/make_judge_figure.py
```

The figures are written to `figures/` as PNG and PDF files.

| Manuscript figure | Content | Script |
| --- | --- | --- |
| Figure 1 | Study design and historical manuscripts | `further_trials/make_figure1_v2.py` |
| Figure 2 | 58 trials published in 2026 and control reviews | `further_trials/run_final_analysis.sh` |
| Supplementary Figure 1 | Judge agreement and instruction checks | `make_judge_figure.py` |
| Supplementary Figure 2 | Scores by manuscript | `make_figures.py` |
| Supplementary Figure 3 | Memory probes | `make_figures.py` |
| Supplementary Figures 4 to 6 | Detection of each error in each of the 58 trials, per model | `further_trials/make_heatmap_parts.py` |
| Supplementary Figure 7 | Cost and review length | `make_figures.py` |
| Supplementary Figure 8 | Subset of three trials rerun through the further-trials pipeline | `further_trials/compare_paper.py` |

The scripts in `code/further_trials/` write to `work_further_trials/figures/`. `make_figures.py` also writes the main figures of the first version (three 2026 trials) as `Figure1_three_trials` and `Figure2_three_trials`. Colours: slate, proprietary models; rose, open-weight models; grey, human reviewers.

See the [code guide](code/README.md) for statistical analyses, validation, study-material scripts and session runners. The recorded prompts are in [`prompts/`](prompts/).

## License and citation

Code is MIT licensed. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff). The test manuscripts are not included because of the licences of the source articles. The scripts that inserted the errors are available from the corresponding author on request.
