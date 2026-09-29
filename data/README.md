# Dataset

## Student Performance — Mathematics

File:
`student-mat.csv`

Dataset:
**UCI Student Performance**

Expected shape:
- Rows: 395
- Columns: 33
- Target: `G3`
- Target scale: 0–20
- Separator: semicolon (`;`)

## Download

From the project root, run:

```bash
python download_dataset.py
```

The script downloads the official UCI archive and extracts:

```text
data/student-mat.csv
```

Official UCI dataset page:

https://archive.ics.uci.edu/dataset/320/student+performance

Official archive:

https://archive.ics.uci.edu/static/public/320/student.zip

## Load in Python

```python
import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=";")

print(df.shape)
print(df.head())
```

The original UCI CSV is semicolon-separated, so `sep=";"` is required.
