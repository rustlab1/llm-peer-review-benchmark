#!/bin/bash
# Analyses and figures for the 58 trials published in 2026 (Figure 2, Supplementary Figs 4 to 6 and 8, and the
# numbers in the text), from saved reviews and judge outputs. No model calls.
#   export SD1=/path/to/Supplementary_Data_1      # extracted archive
#   export FT_WORK=work_further_trials            # made by prepare_workspace.py
#   bash code/further_trials/run_final_analysis.sh
# Run from the repository root. With 10,000 bootstrap resamples this takes about an hour; NBOOT=200 gives the same
# point estimates in a few minutes.
set -e
: "${SD1:?set SD1 to the extracted Supplementary Data 1}"
: "${FT_WORK:?set FT_WORK to the folder made by prepare_workspace.py}"
export SD1 FT_WORK
D=code/further_trials
H=$SD1/data_original/results.json
O="--with-original --orig-root $SD1"
A="python3 $D/analyze_ext.py --human-per-error $H --out $FT_WORK/figures"
ACT="haiku45 sonnet5 gpt54mini gpt54 gpt55 gpt61sol gpt6luna gemini38fl kimik3 dsv4pro qwen38_27b gptoss120b"
# Figure 2: all 15 models; papers N, I and M from the main experiment (four reviews averaged per trial) plus the
# 55 further trials for the 12 models that reviewed them; Opus 5, Fable 5 and Opus 5.5 on N, I and M only (dagger)
$A --trial-set new $O --tag fig2_combined
# Supplementary Figs 4 to 6: per-model heatmaps (12 models, 58 trials)
$A --trial-set new $O --models $ACT --tag suppl_12models
python3 $D/make_heatmap_parts.py
# the 55 further trials only, and the two searches separately (sensitivity analysis)
$A --trial-set new --tag new_trials
$A --set recent --tag recent2026
$A --set phase3 --tag phase3
# papers N, I and M rerun through the same pipeline as the further trials
$A --trial-set rerun --tag rerun_NIM
# sensitivity analysis: without the 14 trials flagged during preparation or reporting a secondary analysis
$A --trial-set new --exclude-noted --tag new_trials_sens_clean
# Supplementary Fig. 8: main experiment vs rerun vs further trials
python3 $D/compare_paper.py "$SD1/data/results_v2.json" new_trials rerun_NIM
# numbers in the text, Figure 1 and the final Figure 2
python3 $D/v2_stats.py
python3 $D/make_figure1_v2.py
python3 $D/make_fig2_only.py
echo "results in $FT_WORK/data, figures in $FT_WORK/figures"
