import csv
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


# opening the CSV file
train_df=pd.read_csv('../DataCollection-RED-Largedataset-Training/OGP_dataset_collection_RED.csv', names=['image_name', 'quarternion_x', 'quarternion_y', 'quarternion_z','quarternion_w','euler_z','euler_y','euler_x','position_X', 'position_Y', 'position_Z'], header=None)


#FINDING MAX AND MIN
for i, pd_row in train_df.iterrows():
	#print(str(pd_row['image_name'])+'.png')

	r = R.from_quat([   pd_row['quarternion_x'],pd_row['quarternion_y'],pd_row['quarternion_z'],pd_row['quarternion_w']   ])
	#print([   pd_row['quarternion_x'],pd_row['quarternion_y'],pd_row['quarternion_z'],pd_row['quarternion_w']   ])
	Rmatrix=r.as_matrix()
	print(Rmatrix)
	rows = [ str(pd_row['image_name']),Rmatrix[0][0],Rmatrix[0][1],Rmatrix[0][2],Rmatrix[1][0],Rmatrix[1][1],Rmatrix[1][2],pd_row['position_X'],pd_row['position_Y'],pd_row['position_Z']]
	with open('../DataCollection-RED-Largedataset-Training/OGP_dataset_collection_Rmatrix_RED.csv', 'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(rows)



#r = R.from_quat([0.9455291465892524, -0.0543611585918914, 0.1858662478043812, -0.2616739102659405])
#print(r.as_matrix())
#r1 = R.from_quat([0.9455291465892524, -0.0543611585918914, 0.1858662478043812, 0.2616739102659405])
#print(r1.as_matrix())
