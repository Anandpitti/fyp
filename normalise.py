import Image
import sys
from os import listdir
from os.path import isfile, join

from_dir = "/home/anandpitti/Desktop/fyp/crawler/cropped/"
to_dir = "/home/anandpitti/Desktop/fyp/crawler/normalised_dimens_100*100/"

all_files = [f for f in listdir(from_dir + sys.argv[1]) if isfile(join(from_dir + sys.argv[1] + "/" + f))]

count = 0;
for single_img in all_files:
	count+=1;
	image  = Image.open(from_dir + sys.argv[1] + "/" + single_img).convert('RGB')
	thumb = image.resize((100, 100), Image.ANTIALIAS)
	thumb.save(to_dir + sys.argv[1] + "/im" + str(count) + ".jpg")

