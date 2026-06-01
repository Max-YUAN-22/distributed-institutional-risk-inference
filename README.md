# Distributed Institutional Risk Inference

Reproducibility package for the manuscript

> **A Distributed Data Fusion and Lightweight Deep Learning Framework for
> Hierarchical Risk Prediction over Heterogeneous Institutional Data Streams**

submitted to *Cluster Computing*.

The repository contains the implementation of the five-phase distributed
pipeline (per-source extraction, partition-parallel stream merge, idempotent
tensor construction, hierarchical labelling, data-parallel training), the
lightweight LSTM-Attention inference engine (≈ 345 K parameters,
4.7 MB checkpoint), all baseline models, external-validation scripts for the
two public benchmarks (OULAD and EdNet-KT1), the synthetic
institutional-data generator, and the synthetic scale-out stress-test
harness.

The private institutional dataset is **not** included due to data-governance
restrictions; the synthetic generator preserves the feature schema,
marginal ranges, temporal structure, and class-imbalance pattern needed to
exercise the pipeline.

---

## Repository structure

```
distributed-institutional-risk-inference/
├── README.md
├── LICENSE                            # MIT
├── requirements.txt
├── reproduce_public.sh                # one-shot public-benchmark reproduction
├── config/
│   ├── institutional_schema.yaml      # feature schema, ranges, types
│   └── external_benchmark_config.yaml # OULAD / EdNet column maps
├── src/
│   ├── extractors/                    # Phase 1: per-source extractors
│   │   ├── base.py                    # BaseExtractor interface contract
│   │   ├── ams_extractor.py           # Academic Management System
│   │   ├── iucp_extractor.py          # Industry-University Coop Platform
│   │   ├── lms_extractor.py           # Learning Management System
│   │   └── sas_extractor.py           # Student Affairs System
│   ├── fusion/                        # Phases 2-3: merge + tensor construction
│   │   ├── stream_merger.py           # partition-parallel merge on (student, semester)
│   │   ├── tensor_constructor.py      # (N, T, F) tensor with mask
│   │   └── manifest_manager.py        # write-temp / rename-on-success + manifest gate
│   ├── models/                        # Phase 5: inference engine
│   │   ├── lstm_attention_tf.py       # TensorFlow 2.10 implementation
│   │   ├── lstm_attention_torch.py    # PyTorch 1.13 implementation
│   │   └── baselines.py               # LR, RF, XGBoost, vanilla LSTM, GRU
│   ├── evaluation/
│   │   ├── metrics.py                 # accuracy, macro-F1, kappa, per-class P/R/F1
│   │   ├── external_oulad.py          # OULAD preprocessing + early-window cut
│   │   └── external_ednet.py          # EdNet-KT1 preprocessing
│   └── explainability/
│       └── shap_analysis.py           # SHAP feature & category attribution
├── scripts/
│   ├── run_institutional_pipeline.py  # requires the private dataset
│   ├── run_oulad_validation.py        # public, reproducible
│   ├── run_ednet_validation.py        # public, reproducible
│   ├── run_synthetic_scaleout.py      # synthetic-cohort stress test
│   └── generate_figures.py            # all manuscript figures
├── synthetic_data/
│   ├── generate_synthetic_institutional_data.py
│   └── sample_schema.csv
└── tests/
    ├── test_extractors.py
    ├── test_fusion.py
    ├── test_manifest.py
    └── test_metrics.py
```

---

## Reproducibility matrix

| Component                               | Reproducible from this repo? | Notes                                              |
|-----------------------------------------|------------------------------|----------------------------------------------------|
| Institutional training (Tables 2–5)     | Partial                      | Requires the private dataset; schema reproducible. |
| Synthetic scale-out (Table 11)          | **Yes**                      | Uses synthetic generator + local-worker harness.   |
| OULAD external validation (Table 9)     | **Yes**                      | Public dataset.                                    |
| EdNet-KT1 external validation (Table 9) | **Yes**                      | Public dataset.                                    |
| Figures 1–2 (architecture)              | **Yes**                      | Source in `scripts/generate_figures.py`.           |
| Figures 4–8 (results)                   | **Yes**                      | From synthetic + external outputs.                 |
| SHAP attribution (Figure 7)             | **Yes**                      | On synthetic + external models.                    |

---

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the public-benchmark reproduction
bash reproduce_public.sh

# 3. (optional) Synthetic institutional-data generation
python synthetic_data/generate_synthetic_institutional_data.py \
    --n_students 4000 --n_semesters 4 \
    --output synthetic_data/synthetic_4k.csv

# 4. (optional) Synthetic scale-out stress test
python scripts/run_synthetic_scaleout.py \
    --workers 1 2 4 8 --cohort_sizes 4000 16000 64000 100000
```

`reproduce_public.sh` downloads OULAD and EdNet-KT1, runs the early-window
preprocessing pipeline described in §2.6 of the manuscript, trains the
LSTM-Attention model on the 13-feature generic backbone, and reports
the metrics in Table 9.

---

## Environment

The numbers reported in the manuscript were obtained with:

- Python 3.9
- TensorFlow 2.10 (CUDA 11.2) **or** PyTorch 1.13 (CUDA 11.7)
- Single NVIDIA Tesla V100 (32 GB)
- 32-core Intel Xeon E5-2680v4 (for CPU latency measurements)

`requirements.txt` pins the public-benchmark dependencies. Re-running on a
different GPU (e.g., A100, T4) produces metrics within ±0.5 percentage
points of the reported figures; throughput numbers scale with the new
hardware.

---

## Data availability

- **OULAD** — Open University Learning Analytics Dataset, available at
  <https://analyse.kmi.open.ac.uk/open_dataset>.
- **EdNet-KT1** — available at
  <https://github.com/riiid/ednet>.
- **Institutional dataset** — not publicly released. Collected under formal
  data-governance authorisation at Dongying Vocational College; only
  aggregated, fully de-identified records were used.

---

## Citation

If you use this code, please cite the manuscript (BibTeX entry will be
added when the DOI is assigned). The repository will be archived to
Zenodo with a DOI upon publication.

---

## License

MIT — see [LICENSE](LICENSE).
