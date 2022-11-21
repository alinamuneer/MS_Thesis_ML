import pandas as pd

df=pd.read_csv('../DataCollection-REDfirst/OGP_dataset_collection_RED.csv', names=['image_name', 'x', 'y', 'z','w','X', 'Y', 'Z'], header=None)



#FINDING MAX AND MIN
p=df['x'].max()
q=df['x'].min()

print('max min of x')
print(p,q)

p=df['y'].max()
q=df['y'].min()

print('max min of y')
print(p,q)

p=df['z'].max()
q=df['z'].min()

print('max min of z')
print(p,q)

p=df['w'].max()
q=df['w'].min()

print('max min of w')
print(p,q)

p=df['X'].max()
q=df['X'].min()

print('max min of X')
print(p,q)

p=df['Y'].max()
q=df['Y'].min()

print('max min of Y')
print(p,q)

p=df['Z'].max()
q=df['Z'].min()

print('max min of Z')
print(p,q)


