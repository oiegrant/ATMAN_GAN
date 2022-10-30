
# First need to get the statistics on pixel dimensions from scraped images
# 
from PIL import Image
import os

#Get filepaths of all images in img/ folder
#MAKE A FUNC
file_names = []
pathway = 'image_scrapper/imgs'
with os.scandir(pathway) as entries:
    for entry in entries:
        file_names.append(pathway+'/'+entry.name)


def get_num_pixels(path_array):
    dim_array = []
    for path in path_array:
        width, height = Image.open(path).size
        dim_array.append([width, height])
    return (dim_array)

print(get_num_pixels(file_names))

