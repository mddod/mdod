# mdod

MDOD, Multi-Dimensional data Outlier Detection

Python library for Multi-Dimensional data Outlier/Anomaly Detection algorithm.

# MDOD paper

MDOD paper is published in ICAIIC 2024 as title "Outlier Detect using Vector Cosine Similarity by Adding a Dimension"

https://doi.org/10.1109/ICAIIC60209.2024.10463442

# Installation:

<code>pip install mdod</code>

or

<code>git clone https://github.com/mddod/mdod.git</code>

<code>cd mdod</code>

<code>python setup.py install</code>

# usage example:

Please visit https://pypi.org/project/mdod/, or https://github.com/mddod/mdod, or https://mddod.github.io/

Please try testmdodmodelv3.py to see usage


Please try testmdod_simple_example_en.py and testmdod_simple_example_input_data.csv to see usage with reading data from csv file.


<code>PS D:\\mdod> python testmdodmodelv3.py</code>

<code>Generate data: 1000 Sample, 2 Dimension, 150 Outliers</code>

<code>Generated data has been saved to  'generated\_data.csv'</code>

<code>Use sampling rate: 0.05</code>



<code>&nbsp;MDOD Decision Score (Top 10):</code>

<code>\[-19.43729592 -19.37395193 -19.4863282  -18.59004489 -19.54782871</code>

<code>&nbsp;-19.61651847 -19.54449219 -19.49876074 -19.39565567 -19.32848438]</code>

<code>MDOD Predicted Labels (Number of Anomalies): 150</code>

<code>MDOD AUC: 1.0</code>

<code>MDOD Runtime: 0.004329 seconds</code>

<code>MDOD Confusion Matrix:</code>

<code>\[\[850   0]</code>

<code>&nbsp;\[  0 150]]</code>

<code>MDOD Precision: 1.0000</code>

<code>MDOD Recall: 1.0000</code>

<code>MDOD F1-Score: 1.0000</code>



<code>LOF Decision Score (Top 10):</code>

<code>\[1.02516737 0.99665944 0.98142123 1.11726486 0.97569838 0.97965565</code>

<code>&nbsp;1.0517925  0.9823431  0.99890437 1.01750076]</code>

<code>LOF Predicted Labels (Number of Anomalies): 150</code>

<code>LOF AUC: 1.0</code>

<code>LOF Running time: 0.012511 seconds</code>

<code>LOF Confusion Matrix:</code>

<code>\[\[850   0]</code>

<code>&nbsp;\[  0 150]]</code>

<code>LOF Precision: 1.0000</code>

<code>LOF Recall: 1.0000</code>

<code>LOF F1-Score: 1.0000</code>



<code>Comparison results:</code>

<code>Spearman correlation coefficient: 0.9191 (p-value: 0.0000)</code>

<code>MDOD AUC: 1.0000, LOF AUC: 1.0000</code>

<code>Algorithm Complexity Comparison (Empirical Running Time)：MDOD 0.004329s vs LOF 0.012511s</code>



<code>Test Data Comparison (Top 10 Rows):</code>

<code>&nbsp;  Feature\_1  Feature\_2  True\_Label  MDOD\_Score  MDOD\_Label  LOF\_Score  LOF\_Label</code>

<code>0  -0.275279   0.119701         0.0  -19.437296           0   1.025167          0</code>

<code>1  -0.198811   0.218055         0.0  -19.373952           0   0.996659          0</code>

<code>2   0.194494  -0.143846         0.0  -19.486328           0   0.981421          0</code>

<code>3   0.392240  -0.187195         0.0  -18.590045           0   1.117265          0</code>

<code>4  -0.125494  -0.073953         0.0  -19.547829           0   0.975698          0</code>

<code>5  -0.060435   0.074442         0.0  -19.616518           0   0.979656          0</code>

<code>6  -0.155111  -0.310734         0.0  -19.544492           0   1.051793          0</code>

<code>7  -0.005483   0.059862         0.0  -19.498761           0   0.982343          0</code>

<code>8   0.022405   0.247428         0.0  -19.395656           0   0.998904          0</code>

<code>9  -0.252083  -0.098266         0.0  -19.328484           0   1.017501          0</code>

<code>The complete data has been saved to 'test\_data\_comparison.csv'</code>

<code>Test output data has been saved to 'test\_output\_data.csv'</code>

<code>libpng warning: iCCP: cHRM chunk does not match sRGB</code>

<code>libpng warning: iCCP: cHRM chunk does not match sRGB</code>


D:\mdodtest>

D:\mdodtest>

D:\mdodtest>

<code>D:\mdodtest>py testmdod_simple_example_en.py</code>

<code>Number of features: 2</code>

<code>Number of samples: 1000</code>

<code>Number of outliers: 150</code>

<code>Data saved to: testmdod_simple_example_output_data.csv</code>

<code>Sampling rate used: 0.05</code>

<code>MDOD runtime: 0.0797 seconds</code>


<code>MDOD decision scores (top 10 - highest values):</code>

<code>[-6.62289984 -6.77322132 -6.92891522 -7.18871113 -7.23625849 -7.24312221</code>

<code> -7.31773667 -7.32676678 -7.33133378 -7.33982723]</code>

 

<code>MDOD decision scores (bottom 10 - lowest values):</code>

<code>[-19.84866083 -19.82038598 -19.82010554 -19.80062169 -19.80051546</code>

<code> -19.76201034 -19.76181435 -19.75729718 -19.75229983 -19.74520579]</code>

<code></code><br>
<code></code><br>
The <code>parameters of MDOD</code> include: <br> 

<code>norm_distence</code>: The distance of the virtual dimension (the default is 1.0, which affects the similarity calculation).<br> 

<code>top_n</code>: Consider the number of nearest points used for statistics (select the appropriate number for outlier identification and distinction).<br>

<code>Contamination</code>: The set abnormal ratio (the default is 0.1, which is used for threshold calculation).<br> 

<code>sampling_rate</code>: sampling rate (0~1, default 1.0, low value accelerates calculation).<br> 

<code>random_state</code>: random seed.<br> <br>  







