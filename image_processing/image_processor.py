
# First need to get the statistics on pixel dimensions from scraped images
# 
from PIL import Image
import os
import statistics


#Get filepaths of all images in img/ folder
#MAKE A FUNC
file_names = []
pathway = '../ATMAN_GAN/image_scrapper/imgs'
with os.scandir(pathway) as entries:
    for entry in entries:
        file_names.append(pathway+'/'+entry.name)


def get_num_pixels(path_array):
    dim_array = []
    for path in path_array:
        width, height = Image.open(path).size
        dim_array.append([width, height])
    return (dim_array)



dim_array = get_num_pixels(file_names)

print(len(dim_array))
print([item[0] for item in dim_array])
width_avg = statistics.mean([item[0] for item in dim_array])
height_avg = statistics.mean([item[1] for item in dim_array])


