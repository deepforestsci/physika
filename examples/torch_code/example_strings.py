import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
empty_string = ''
name = 'Physika'
msg = 'Hello, World!'
print(print(empty_string))
print(print(name))
print(print(msg))
first_name = 'Physika'
last_name = 'Language'
full_name = ((first_name + ' ') + last_name)
print(print(full_name))
first_character = name[int(0)]
third_character = name[int(2)]
print(print(first_character))
print(print(third_character))
first_part = full_name[:int(7)]
last_part = full_name[int(8):]
print(print(first_part))
print(print(last_part))