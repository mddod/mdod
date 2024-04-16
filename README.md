# mdod
MDOD, Multi-Dimensional data Outlier Detection

Python library for Multi-Dimensional data Outlier/Anomaly Detection algorithm.

# Installation:
 pip install mdod

or

 git clone https://github.com/mddod/mdod.git

 cd mdod

 python setup.py install

# usage example:
import mdod

localFile = 'TestDataset.txt'

dets= np.loadtxt(localFile,delimiter=',')

nd = 1

sn = 15

result = mdod.md(dets,nd,sn)

print (result)

# TestDataset.txt format:
data1,data2,data3,data4,data5,data6

data1,data2,data3,data4,data5,data6

data1,data2,data3,data4,data5,data6

...

# dets format:
[[data1 data2 data3 data4 data5 data6] 

[data1 data2 data3 data4 data5 data6] 

[data1 data2 data3 data4 data5 data6] 

...]

# rusult format:
[value1, '[data1 data2 data3 data4 data5 data6]', '0']

[value2, '[data1 data2 data3 data4 data5 data6]', '1']

[value3, '[data1 data2 data3 data4 data5 data6]', '2']

...

# MDOD paper 
MDOD paper is published in ICAIIC 2024 as title "Outlier Detect using Vector Cosine Similarity by Adding a Dimension" 

https://doi.org/10.1109/ICAIIC60209.2024.10463442

 
 
