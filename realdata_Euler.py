import csv
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


# opening the CSV file
test_df=pd.read_csv('../DataCollection-Real-Pr2/ogp_real_world.csv', names=['image_name', 'position_X', 'position_Y', 'position_Z','quarternion_x', 'quarternion_y', 'quarternion_z','quarternion_w'], header=None)

#FINDING MAX AND MIN
for i, pd_row in test_df.iterrows():
	print(str(pd_row['image_name'])+'.png')
	
	r = R.from_quat([   pd_row['quarternion_x'],pd_row['quarternion_y'],pd_row['quarternion_z'],pd_row['quarternion_w']   ])
	euler=r.as_euler('zyx', degrees=True)
	rows = [ str(pd_row['image_name'])+'.png',euler[0],euler[1],euler[2] ,pd_row['position_X'],pd_row['position_Y'],pd_row['position_Z']]
	with open('../DataCollection-Real-Pr2/ogp_real_world_Euler.csv', 'a') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(rows)
