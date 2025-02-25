from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
data = {
    'a':[1,2,3,4,5],
    'b':[2,4,6,8,10],
}
df=pd.DataFrame(data)

X=df[['a']]

Y=df[['b']]

X_train,Y_train,Y_test,X_test=train_test_split(X,Y,test_size =0.2,random_state=42)
