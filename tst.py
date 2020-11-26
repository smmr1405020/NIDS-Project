import os

files = os.listdir('models/leaf_models')
for i in files:
    str_i = str(i).split("_")
    print(str_i[1])