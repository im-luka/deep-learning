import numpy as np
import time

result_file = "labos_files/result_file.npy"

start_time = time.time()
with open(result_file, "rb") as file:
  matrix = np.load(file)

print(matrix)
print('Time needed for reading matrix from file %s' % (time.time() - start_time))
print(matrix.shape)