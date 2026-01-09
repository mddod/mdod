# MDOD - Multi-Dimensional Outlier Detection

**MDOD** is a Python library for outlier detection in multi-dimensional data using vector cosine similarity with an added virtual dimension.

- v3.0.6.1 (High-Performance & Large-Scale) Features:
  Fully backward compatible with published API. 
  Supports >500k samples without OOM (batch + float32 + incremental). 
  Preserves high AUC by only down-sampling when absolutely necessary. 
  MemoryError auto-fallback (rarely triggered). 

- The **core algorithm** is based on the author's original academic paper:  
  *"Outlier Detection Using Vector Cosine Similarity by Adding a Dimension"*  
  DOI: [10.48550/arXiv.2601.00883](https://doi.org/10.48550/arXiv.2601.00883)

- The **code implementation** was developed by the author with significant optimization assistance from Grok (xAI).

This software is licensed under the **BSD 3-Clause License** - see the [LICENSE.txt](LICENSE.txt) file for details.

**Disclaimer**  
This library is provided "AS IS", WITHOUT ANY WARRANTY of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are strongly advised to thoroughly test and validate the library for their specific use case, especially in production environments or critical applications.

## Installation

```bash
pip install mdod
```
Or from source:
```bash
git clone https://github.com/mddod/mdod.git
cd mdod
python setup.py install
```

## Usage Example

Please visit the repository for detailed examples:  
https://github.com/mddod/mdod  

or the documentation site: https://mddod.github.io/

You can also run `testmdodmodelv3.py` or `testmdod_simple_example_en.py` to see demonstrations.

### Example Performance (Synthetic Dataset: 1000 samples, 2D, 150 outliers, sampling_rate=0.05)

- MDOD AUC: 1.0
- MDOD Runtime: ~0.009 seconds (significantly faster than LOF)
- Spearman correlation with LOF scores: 0.9192


## Parameters

- `norm_distance`: Distance of the virtual dimension (default: 1.0) — affects similarity sensitivity.
- `top_n`: Number of top similar points to consider (default: 5).
- `contamination`: Expected proportion of outliers (default: 0.1) — used for threshold.
- `sampling_rate`: Sampling ratio (0–1, default: 1.0) — lower values speed up computation.
- `random_state`: Random seed for reproducibility.

## Performance Comparison with LOF

On standard test datasets, MDOD consistently achieves perfect or near-perfect detection (AUC ≈ 1.0) while being **2–3x faster** than scikit-learn's Local Outlier Factor (LOF).


## Quick Start

```python
from mdod import MDOD
import numpy as np

# Example data (replace with your own dataset)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features

model = MDOD(
    norm_distance=1.0,
    top_n=5,
    contamination=0.1,
    sampling_rate=0.1,  # Use lower values for faster computation on large data
    random_state=42
)
model.fit(X)

# Predict outliers
labels = model.predict()                  # 1 = outlier, 0 = normal
scores = model.decision_function(X)       # indicate outlier probability
```