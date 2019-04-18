
## Objectives
YWBAT 
* write functions to calcualte statistics and apply those functions to a dataframe
* write functions to plot distributions
* interpret data based on the statistics
* remove outliers from data by using pandas slicing
* use pandas to create new columns


```python
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from pprint import pprint

import matplotlib.pyplot as plt
```

### always do this step first when loading in a pre built dataset


```python
wine = load_wine()
```


```python
data = wine.data
target = wine.target

columns = wine.feature_names + ["target"]
```


```python
df = pd.DataFrame(np.column_stack([data, target]), columns=columns)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Write a function


```python
### write a function that takes in a load method 
### from sklearn datasets and returns a dataframe
### like the one above
def load_dataset(load_set = None):
    pass
```

### Let's look at a description


```python
pprint(wine.DESCR)
```

    ('.. _wine_dataset:\n'
     '\n'
     'Wine recognition dataset\n'
     '------------------------\n'
     '\n'
     '**Data Set Characteristics:**\n'
     '\n'
     '    :Number of Instances: 178 (50 in each of three classes)\n'
     '    :Number of Attributes: 13 numeric, predictive attributes and the class\n'
     '    :Attribute Information:\n'
     ' \t\t- Alcohol\n'
     ' \t\t- Malic acid\n'
     ' \t\t- Ash\n'
     '\t\t- Alcalinity of ash  \n'
     ' \t\t- Magnesium\n'
     '\t\t- Total phenols\n'
     ' \t\t- Flavanoids\n'
     ' \t\t- Nonflavanoid phenols\n'
     ' \t\t- Proanthocyanins\n'
     '\t\t- Color intensity\n'
     ' \t\t- Hue\n'
     ' \t\t- OD280/OD315 of diluted wines\n'
     ' \t\t- Proline\n'
     '\n'
     '    - class:\n'
     '            - class_0\n'
     '            - class_1\n'
     '            - class_2\n'
     '\t\t\n'
     '    :Summary Statistics:\n'
     '    \n'
     '    ============================= ==== ===== ======= =====\n'
     '                                   Min   Max   Mean     SD\n'
     '    ============================= ==== ===== ======= =====\n'
     '    Alcohol:                      11.0  14.8    13.0   0.8\n'
     '    Malic Acid:                   0.74  5.80    2.34  1.12\n'
     '    Ash:                          1.36  3.23    2.36  0.27\n'
     '    Alcalinity of Ash:            10.6  30.0    19.5   3.3\n'
     '    Magnesium:                    70.0 162.0    99.7  14.3\n'
     '    Total Phenols:                0.98  3.88    2.29  0.63\n'
     '    Flavanoids:                   0.34  5.08    2.03  1.00\n'
     '    Nonflavanoid Phenols:         0.13  0.66    0.36  0.12\n'
     '    Proanthocyanins:              0.41  3.58    1.59  0.57\n'
     '    Colour Intensity:              1.3  13.0     5.1   2.3\n'
     '    Hue:                          0.48  1.71    0.96  0.23\n'
     '    OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71\n'
     '    Proline:                       278  1680     746   315\n'
     '    ============================= ==== ===== ======= =====\n'
     '\n'
     '    :Missing Attribute Values: None\n'
     '    :Class Distribution: class_0 (59), class_1 (71), class_2 (48)\n'
     '    :Creator: R.A. Fisher\n'
     '    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n'
     '    :Date: July, 1988\n'
     '\n'
     'This is a copy of UCI ML Wine recognition datasets.\n'
     'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data\n'
     '\n'
     'The data is the results of a chemical analysis of wines grown in the same\n'
     'region in Italy by three different cultivators. There are thirteen '
     'different\n'
     'measurements taken for different constituents found in the three types of\n'
     'wine.\n'
     '\n'
     'Original Owners: \n'
     '\n'
     'Forina, M. et al, PARVUS - \n'
     'An Extendible Package for Data Exploration, Classification and '
     'Correlation. \n'
     'Institute of Pharmaceutical and Food Analysis and Technologies,\n'
     'Via Brigata Salerno, 16147 Genoa, Italy.\n'
     '\n'
     'Citation:\n'
     '\n'
     'Lichman, M. (2013). UCI Machine Learning Repository\n'
     '[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,\n'
     'School of Information and Computer Science. \n'
     '\n'
     '.. topic:: References\n'
     '\n'
     '  (1) S. Aeberhard, D. Coomans and O. de Vel, \n'
     '  Comparison of Classifiers in High Dimensional Settings, \n'
     '  Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  \n'
     '  Mathematics and Statistics, James Cook University of North Queensland. \n'
     '  (Also submitted to Technometrics). \n'
     '\n'
     '  The data was used with many others for comparing various \n'
     '  classifiers. The classes are separable, though only RDA \n'
     '  has achieved 100% correct classification. \n'
     '  (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) \n'
     '  (All results using the leave-one-out technique) \n'
     '\n'
     '  (2) S. Aeberhard, D. Coomans and O. de Vel, \n'
     '  "THE CLASSIFICATION PERFORMANCE OF RDA" \n'
     '  Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of \n'
     '  Mathematics and Statistics, James Cook University of North Queensland. \n'
     '  (Also submitted to Journal of Chemometrics).\n')



```python
### calculate the mean of malic acid without using .mean() or np.mean()
mean_malic_acid = None
mean_malic_acid
```


```python
### calculate the standard deviation of malic acid without using .std() or np.std()
std_malic_acid = None
std_malic_acid
```


```python
### write functions that takes in a list and returns the mean and standard deviation
def mean_function():
    pass

def std_function():
    pass
```


```python
### test your function against a different column in the dataframe
column = None
mean_function(df[column]) == df[column].mean(), std_function(df[column]) == df[column].std()
```

### Plot Distribution


```python
### plot the distribution of alcohol content
### code goes here

```


```python
### write a function that takes in a dataframe and column name and returns a histogram
### the histogram must have labeled axes and a title

def histogram():
    pass
```

### Creating columns with pandas


```python
### create a new column that is the ratio of ash to malic acid and name it
### ash_to_malic_acid

# code goes here
```
