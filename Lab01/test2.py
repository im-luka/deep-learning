import numpy as np
import time

test_file = "test/test_file.txt"
result_file = "test/result_file.npy"

with open(test_file, "r") as file:
  book = file.read()
print(len(book))

chars = [char for char in book if char.isalnum()]
print("Lista slova", len(chars), chars)
set_of_chars = set(chars)
print("Set slova", len(set_of_chars), set_of_chars)

chars_list = []
i = 0
while i < len(chars):
  helper = chars[i:i+10]
  if(len(helper) == 10):
    chars_list.append(helper)
  i += 5
print(chars_list)

char_index = dict((char, index) for index, char in enumerate(set_of_chars))
index_char = dict((index, char) for index, char in enumerate(set_of_chars))

# ndarray(n, 50, m)
matrix_n = len(chars_list) # ukupan broj podlisti
matrix_m = len(set_of_chars) # ukupan broj unikatnih slova

print("N velicina matrice (broj podlisti)", matrix_n)
print("M velicina matrice (broj unikatnih znakova)", matrix_m)

matrix = np.zeros((matrix_n, 10, matrix_m), dtype=np.int32)

print(matrix)

print(chars_list)
print(char_index) 
print('----matrix with for loop (range func)----')
start_time = time.time()
for i in range(len(chars_list)):
  for j in range(len(chars_list[i])):
    char = chars_list[i][j]
    matrix[i][j][char_index[char]] = 1
print(matrix)
print('time needed with for loop range func %s' % (time.time() - start_time))

print(chars_list)
print(char_index) 
print('----matrix with enumerate----')
start_time = time.time()
for i, sublist in enumerate(chars_list):
  for j, element in enumerate(sublist):
    matrix[i][j][char_index[element]] = 1
print(matrix)
print('time needed with enumerate %s' % (time.time() - start_time))

with open(result_file, "wb") as file:
  np.save(file, matrix)

# with open(result_file, "rb") as file:
#   matrix = np.load(file)
# print(matrix)