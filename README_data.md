# Dataset Instructions

The DASS-42 dataset is **not included** in this repository.

## Download

1. Go to: https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses
2. Click **Download** (requires a free Kaggle account)
3. Unzip the downloaded file
4. Place `data.csv` in the **project root directory** (same level as `raas_experiment.py`)

## Dataset Details

- **Source:** Kaggle, uploaded by L. Greenwell (2019)
- **Original collection:** Open-source questionnaire platform, 2017–2019
- **Format:** Tab-separated values (.csv)
- **Size:** ~40,000 responses
- **Item columns:** Q1A through Q42A (four-point Likert: 1–4)
- **License:** CC0 (Public Domain)

## Preprocessing Applied

The following steps are performed automatically by `raas_experiment.py`:

1. Parse with `sep='\t'`, skip malformed rows
2. Coerce item columns to numeric; drop rows with missing item values
3. Filter age to 13–80 years
4. Result: **39,764 valid samples**

## Label Construction

Labels (depression, anxiety, stress) are derived from official DASS-42 subscales
using the 60th-percentile threshold on each subscale sum.

This threshold (depression: 39, anxiety: 32, stress: 38 on the 1–4 scale)
**exceeds** the DASS official moderate lower bounds (depression: 28, anxiety: 24,
stress: 33 on equivalent scale), meaning positive labels correspond to
moderate-to-severe symptomatology.
