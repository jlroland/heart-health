import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import os
from collections import defaultdict

files = os.listdir('data/data_2016/') # dir is your directory path
number_files = len(files)

df_dict = defaultdict()
for file in files:
    df_dict[file.replace('.csv', '')] = pd.read_csv('data/data_2016/{}'.format(file)).set_index('SEQN')

agg_data = pd.concat(df_dict.values(), axis=1)

agg_data.drop(['PEASCCT1', 'BPAARM', 'BPACSZ', 'BPXPULS', 'BPXPTY', 'BPXML1', 'BPAEN1', 'BPAEN2', 'BPAEN3', 'BPAEN4'], axis=1, inplace=True)

