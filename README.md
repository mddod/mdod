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

# usage example testmdod.py (get data from TestDataset.txt):
import numpy as np

import mdod

localFile = 'TestDataset.txt'

dets= np.loadtxt(localFile,delimiter=',')

#nd: value of the observation point in the new dimension

nd = 1

#sn: number of statistics on the first few numbers in the order of scores from large to small

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

# result format:
[value1, '[data1 data2 data3 data4 data5 data6]', '0']

[value2, '[data1 data2 data3 data4 data5 data6]', '1']

[value3, '[data1 data2 data3 data4 data5 data6]', '2']

...

# usage example for radom 2D data:
testmdod2D.py
# usage example for radom 3D data:
testmdod3D.py


# More details:
Pypi mdod library please visit https://pypi.org/project/mdod/
Python examples please visit https://mddod.github.io/

 
 
