import numpy as np
import glob

text_files_dir = "test_files"
result_file = "test_files/result_file.npy"

all_text = ""
for input_file in glob.glob(text_files_dir + "/*.txt"):
  with open(input_file, "r", encoding="utf-8") as file:
    all_text += file.read()
print(all_text)

chars = [char for char in all_text if char.isalnum()]
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

for i in range(len(chars_list)):
  for j in range(len(chars_list[i])):
    char = chars_list[i][j]
    matrix[i][j][char_index[char]] = 1

print(chars_list)
print(char_index) 
print('----final matrix----')
print(matrix)

with open(result_file, "wb") as file:
  np.save(file, matrix)
