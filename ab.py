import pandas as pd
data={
    'student':['mohan','imran','akash','andy','jack'],
    'maths':[90,85,95,88,92],
    'science':[78,92,88,76,89],
    'english':[92,89,95,88,90]
}

df=pd.DataFrame(data)
print(df)

#mean
subject_mean =df[['maths','science','english']].mean()

print(subject_mean)

subject_median=df[['maths','english','science']].median()
print(subject_median)

df['total']=df[['maths','science','english']].sum(axis=1)

top_scorer=df.loc[df['total'].idxmax(),'student']
print(top_scorer)