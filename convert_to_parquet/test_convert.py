import numpy as np
import os
from pyspark.ml.linalg import Vectors

npz_files_path = "/home/lmtruong1512/Codes/BTL_CSDLDPT/extracted_files/extracted_SIFT100"

file_names = os.listdir(npz_files_path)[0:10]
np_arrs = [np.load(os.path.join(npz_files_path, file_name))['arr_0']
           for file_name in file_names]
dataset = map(lambda x: Vectors.dense(x,), np_arrs)
print(dataset)
