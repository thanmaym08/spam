import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples=300
outliner_fraction=0.05
x_inliner=np.random.normal(0,1,size=(n_samples,2))
x_outliner=np.random.uniform(-4,4,size=(int(n_samples*outliner_fraction),2))

