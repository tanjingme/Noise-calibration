""" Evaluation of color bias """

import rawpy
import numpy as np
from util import color_bias_eval
import seaborn as sns

#--------------- Step 1: Data Process ---------------
# Setup: Canon EOS M50
# Configuration: ISO-3200  f/#: f/5.6  f:15mm 
# Temperature: room temperature

# index for the beginning and the end of bias frames
n1, n2 = 82, 165
R_bias, G1_bias, G2_bias, B_bias = color_bias_eval(n1, n2)

#--------------- Step 2: Plot ---------------
sns.set_theme()
sns.histplot(R_bias, stat="proportion", kde=True).set(title='R')
# sns.histplot(G1_bias, stat="proportion", kde=True).set(title='G1')
# sns.histplot(G2_bias, stat="proportion", kde=True).set(title='G2')
# sns.histplot(B_bias, stat="proportion", kde=True).set(title='B')
