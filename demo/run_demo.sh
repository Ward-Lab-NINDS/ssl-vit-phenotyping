#!/usr/bin/env bash
set -euo pipefail
python demo/generate_synthetic_benchmark.py
ssl-vit-benchmark \
  --input demo/synthetic_outputs/synthetic_phenotype_cp_ssl.tsv \
  --output-dir demo/synthetic_outputs/benchmarks \
  --label-col sgRNA \
  --replicate-col replicate \
  --perturbation-col perturbation \
  --control-col control_type \
  --feature-set classical=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_ \
  --feature-set ssl=prefix:ssl_ \
  --feature-set combined=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_,prefix:ssl_ \
  --procode-channel procode_ch1 \
  --procode-channel procode_ch2 \
  --expected-signature 00 \
  --expected-signature 10 \
  --expected-signature 01 \
  --expected-signature 11 \
  --procode-threshold-method otsu \
  --batch-col plate \
  --batch-col well \
  --batch-col density \
  --write-report
