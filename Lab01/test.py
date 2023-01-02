test_file = "test/test_file.txt"

with open(test_file, "r") as file:
  book = file.read()

words = book.split()
print(words)
print('broj rijeci ukupno %s' % len(words))
words = list(set(words))
print('broj rijeci nakon filtriranja %s' % len(words))
print(words)

chars = [char for char in book if char.isalnum()]
print('broj slova ukupno %s' % len(chars))
chars = list(set(chars))
print('broj slova nakon filtriranja %s' % len(chars))
print(chars)

lista = ['v', 't', 'r', 'x', 'T', 'u', 'p', 'o', 'e', 'L', 'g', 'f', 'l', 'c', 'm', 'w', 'b', 'H', 'a', 'n', 'j', 'y', 'd', 'i', 's', 'k', 'h']
print(lista)

lista2 = []
i = 0
while i < len(lista):
  lista2.append(lista[i:i+5])
  i += 3

if(len(lista2[-1]) <= 5):
  lista2.pop(-1)
print(lista2)

test = '123456789465476874168764998422848421659876265955685412678411548781256568232'
test_lista = [char for char in test]
print(test_lista)
original_list = []
i = 0
while i < len(test_lista):
  original_list.append(test_lista[i:i+50])
  i += 5
print(original_list)


