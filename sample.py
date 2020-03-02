import numpy as np
import os
import random
import os.path 
import shutil
import PIL
from PIL import Image
import numpy as np

# ('origin_train len', 16785)
# ('-----------cat_bright', 3086)
# ('-----------dog_bright', 5299)
# ('-----------cat dark', 4748)
# ('-----------dog dark', 3652)
# -------------------------------------
# Train data
# ('sample_cat_bright: ', 925)
# ('sample_dog_bright: ', 4769)
# ('sample_cat_dark: ', 4273)
# ('sample_dog_dark: ', 1095)
# ('All sampled Train data : ', 11062)
# -------------------------------------


test_sample = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/learning-not-to-learn/dataset/dogs_and_cats/sampled/test/'
sampled_test = os.listdir(test_sample)
# cat bright : dog bright = 1 : 9
# cat dark : dog dark = 9 : 1
train_sample = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/learning-not-to-learn/dataset/dogs_and_cats/sampled/train/'
sampled_train = os.listdir(train_sample)
# # original test dataset
# origin_test = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/learning-not-to-learn/dataset/dogs_and_cats/test'
# test = os.listdir(origin_test)
# print('origin_test len',len(test))


# original train dataset
origin_train = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/learning-not-to-learn/dataset/dogs_and_cats/train/'
train = os.listdir(origin_train)
print('origin_train len',len(train))

light_txt = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/learning-not-to-learn/dataset/dogs_and_cats/list_bright.txt'
dark_txt = '/media/doh/ECECE209ECE1CDC0/research/cats_and_dogs/learning-not-to-learn/dataset/dogs_and_cats/list_dark.txt'

with open(light_txt, 'r') as f:
    bright = [line.strip() for line in f]

with open(dark_txt, 'r') as f:
    dark = [line.strip() for line in f]

print(len(bright))
print(len(dark))
print(len(bright)+len(dark))
cat_bright = []
dog_bright = []
cat_dark = []
dog_dark = []
for i in range(len(bright)):
    if 'cat' in bright[i][0:3]:
        cat_bright.append(bright[i])
    else :
        dog_bright.append(bright[i])

for i in range(len(dark)):
    if 'cat' in dark[i][0:3]:
        cat_dark.append(dark[i])
    else :
        dog_dark.append(dark[i])


# sample dark
sample_cat_bright = random.sample(cat_bright, int(0.1*len(cat_bright)))
sample_dog_bright = random.sample(dog_bright, int(0.9*len(dog_bright)))
sample_cat_dark = random.sample(cat_dark, int(0.9*len(cat_dark)))
sample_dog_dark = random.sample(dog_dark, int(0.1*len(dog_dark)))
sample_train = sample_cat_bright + sample_dog_bright + sample_cat_dark + sample_dog_dark
print('sample_cat_bright: ', len(sample_cat_bright)) 
print('sample_dog_bright: ', len(sample_dog_bright))
print('sample_cat_dark: ', len(sample_cat_dark))
print('sample_dog_dark: ', len(sample_dog_dark))
print('All sampled Train data : ', len(sample_train))




# train set
# cat_bright = cat_bright[0:308]
# dog_bright = dog_bright[0:4769]
# cat_dark = cat_dark[0:4273]
# dog_dark = dog_dark[0:365]
# final = cat_bright + dog_bright +cat_dark+dog_dark
# print(sample_train)
# # move original train data to sampled


# # # test set 1:1
# cat_bright = cat_bright[309:609]

# dog_bright = dog_bright[4769:5069]
# cat_dark = cat_dark[4273:4573]
# dog_dark = dog_dark[365:600]
# final_test = cat_bright + dog_bright +cat_dark+dog_dark
# print(len(final_test))

# ('origin_train len', 16785)
# ('-----------cat_bright', 3086)
# ('-----------dog_bright', 5299)
# ('-----------cat dark', 4748)
# ('-----------dog dark', 3652)
# -------------------------------------

# test set 1:9
cat_bright = cat_bright[309:1186]
dog_bright = dog_bright[4769:4839]
cat_dark = cat_dark[4273:4348]
dog_dark = dog_dark[365:1552]
final_test = cat_bright + dog_bright +cat_dark+dog_dark
print(len(final_test))



# print(len(train))
# print(len(sample_train))




print('train : ', len(sampled_train))
# print('test : ', len(sampled_test))
# count = 0
# for i in range(len(train)):
#     if train[i] in final :
#         count = count+1
#         shutil.copy(origin_train + train[i], train_sample)
    
# print(count)

# test 1:1
# count = 0
# for i in range(len(train)):
#     if train[i] in final_test :
#         count = count+1
#         shutil.copy(origin_train + train[i], test_sample)
    
# print(count)

# test same as train
# count = 0
# for i in range(len(train)):
#     if train[i] in final_test :
#         count = count+1
#         shutil.copy(origin_train + train[i], test_sample)
    
# print(count)


# test reverse ratio as train
# count = 0
# for i in range(len(train)):
#     if train[i] in final_test :
#         count = count+1
#         shutil.copy(origin_train + train[i], test_sample)
    
# print(count)


print("done")




