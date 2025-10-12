# mdod

MDOD, Multi-Dimensional data Outlier Detection

Python library for Multi-Dimensional data Outlier/Anomaly Detection algorithm.

# MDOD paper

MDOD paper is published in ICAIIC 2024 as title "Outlier Detect using Vector Cosine Similarity by Adding a Dimension"

https://doi.org/10.1109/ICAIIC60209.2024.10463442

# Installation:

pip install mdod

or

git clone https://github.com/mddod/mdod.git

cd mdod

python setup.py install

# usage example:

Please visit https://github.com/mddod/mdod, or https://mddod.github.io/

Please try testmdodmodelv3.py to see usage



PS D:\\mdod> python testmdodmodelv3.py

Generate data: 1000 Sample, 2 Dimension, 150 Outliers

Generated data has been saved to  'generated\_data.csv'

Use sampling rate: 0.05



&nbsp;MDOD Decision Score (Top 10):

\[-19.43729592 -19.37395193 -19.4863282  -18.59004489 -19.54782871

&nbsp;-19.61651847 -19.54449219 -19.49876074 -19.39565567 -19.32848438]

MDOD Predicted Labels (Number of Anomalies): 150

MDOD AUC: 1.0

MDOD Runtime: 0.004329 seconds

MDOD Confusion Matrix:

\[\[850   0]

&nbsp;\[  0 150]]

MDOD Precision: 1.0000

MDOD Recall: 1.0000

MDOD F1-Score: 1.0000



LOF Decision Score (Top 10):

\[1.02516737 0.99665944 0.98142123 1.11726486 0.97569838 0.97965565

&nbsp;1.0517925  0.9823431  0.99890437 1.01750076]

LOF Predicted Labels (Number of Anomalies): 150

LOF AUC: 1.0

LOF Running time: 0.012511 seconds

LOF Confusion Matrix:

\[\[850   0]

&nbsp;\[  0 150]]

LOF Precision: 1.0000

LOF Recall: 1.0000

LOF F1-Score: 1.0000



Comparison results:

Spearman correlation coefficient: 0.9191 (p-value: 0.0000)

MDOD AUC: 1.0000, LOF AUC: 1.0000

Algorithm Complexity Comparison (Empirical Running Time)：MDOD 0.004329s vs LOF 0.012511s



Test Data Comparison (Top 10 Rows):

&nbsp;  Feature\_1  Feature\_2  True\_Label  MDOD\_Score  MDOD\_Label  LOF\_Score  LOF\_Label

0  -0.275279   0.119701         0.0  -19.437296           0   1.025167          0

1  -0.198811   0.218055         0.0  -19.373952           0   0.996659          0

2   0.194494  -0.143846         0.0  -19.486328           0   0.981421          0

3   0.392240  -0.187195         0.0  -18.590045           0   1.117265          0

4  -0.125494  -0.073953         0.0  -19.547829           0   0.975698          0

5  -0.060435   0.074442         0.0  -19.616518           0   0.979656          0

6  -0.155111  -0.310734         0.0  -19.544492           0   1.051793          0

7  -0.005483   0.059862         0.0  -19.498761           0   0.982343          0

8   0.022405   0.247428         0.0  -19.395656           0   0.998904          0

9  -0.252083  -0.098266         0.0  -19.328484           0   1.017501          0

The complete data has been saved to 'test\_data\_comparison.csv'

Test output data has been saved to 'test\_output\_data.csv'

libpng warning: iCCP: cHRM chunk does not match sRGB

libpng warning: iCCP: cHRM chunk does not match sRGB

> 







