import cv2
import glob
import numpy as np

# imdir = 'Dataset/seledri/'
# ext = ['jpg']

# files = []
# [files.extend(glob.glob(imdir + '*.jpg'))]

# images = [cv2.imread(file)for file in files]

# i = 1
# for image in images:
#     im_edge = cv2.Canny(image,100,200)
#     if i<10:
#         im_name = 'Dataset/seledri_edge/00'+ str(i) + '.jpg'
#     else : 
#         im_name = 'Dataset/seledri_edge/0'+ str(i) + '.jpg'
#     cv2.imwrite(im_name,im_edge)

#     i+=1

imdir = 'Dataset/jeruk nipis/'
ext = ['jpg']

files = []
[files.extend(glob.glob(imdir + '*.jpg'))]

images = [cv2.imread(file)for file in files]

i = 1
for image in images:
    im_edge = cv2.Canny(image,100,200)
    if i<10:
        im_name = 'Dataset/jeruk_nipis_edge/00'+ str(i) + '.jpg'
    else : 
        im_name = 'Dataset/jeruk_nipis_edge/0'+ str(i) + '.jpg'
    cv2.imwrite(im_name,im_edge)

    i+=1