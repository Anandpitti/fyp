
# import the necessary packages
from __future__ import division
from skimage import feature
from numpy import arctan2, fliplr, flipud
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from os import listdir




cluster_list = [[0],
			[1],
			[2],
			[3],
			[4],
			[5],
			[6],
			[7],
			[8],
			[9],
			[10],
			[11],
			[12],
			[13],
			[14],
			[15],
			[16]]

max_clusters = 17

index_to_cluster = [0,
			1,
			2,
			3,
			4,
			5,
			6,
			7,
			8,
			9,
			10,
			11,
			12,
			13,
			14,
			15,
			16]

aaclass_list = ['oxygen_cylinder',
			'x_ray',
			'stethoscope',
			'patients',
			'signs_symbols',
			'surgical_knife',
			'surgical_scissors',
			'surveillance_cameras',
			'tablets_medicine',
			'tablet_strip_back',
			'treadmill_hospital',
			'waiting_room',
			'walker',
			'weight_machine',
			'wheel_chair',
			'wrap_bandage',
			'x_Ray_machine']

class_list = [
			'oxygen_cylinder',
			'surgical_scissors',
			'walker',
			'surveillance_cameras',
			'stethoscope',
			'weight_machine',
			'treadmill_hospital',
			'wheel_chair',
			'surgical_knife',
			'waiting_room',
			'wrap_bandage',
			'x_Ray_machine'
			]

check_cluster_list = [[0, 7, 14, 12, 13, 15], [1, 2], [3, 11, 16, 10, 8, 9], [4, 5, 6]]
check_index_to_cluster = [0, 1, 2, 0, 0, 0, 3, 0, 1, 1, 0, 1]


def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def image_to_feature_vector_hu_moments(imagePath, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	image = cv2.imread(imagePath,0) #Read image
	image_t = cv2.medianBlur(image,7) #Apply filter
	th1 = cv2.adaptiveThreshold(image_t,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,255,2) #Use the adaptive threshold technique to get a binary image from the original
	kernel = np.ones((5,5),np.uint8) #Define structuring element
	img=cv2.dilate(th1,kernel,iterations = 1) #Dilate the binary image
	HM=cv2.HuMoments(cv2.moments(img)).flatten() #Get image features through HuMoments
	return HM

def image_to_feature_vector_hough_transform(imagePath, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	image = cv2.imread(imagePath) #Read image
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)		
	return edges.flatten()

def image_to_feature_vector_harris_corner(imagePath, size=(32, 32)):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)	
	return dst.flatten()

def image_to_feature_vector_st_corner(imagePath, size=(32, 32)):
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
	corners = corners.flatten()
	return corners

def image_to_feature_vector_sift(imagePath, size=(32, 32)):
	img = cv2.imread(imagePath)
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	(kps, descs) = sift.detectAndCompute(gray, None)
	return descs

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()

def extract_hog_histogram(image, bins=(8, 8, 8)):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = imutils.auto_canny(gray)
	H = feature.hog(edged, orientations=9, pixels_per_cell=(10, 10),
	cells_per_block=(2, 2), normalise=True)
	return H

def print_count_each_class(print_name, in_list):
	#checking the number of terms in train class list
	temp_label_list = {}
	for i in in_list:
		if i in temp_label_list:
			temp_label_list[i] += 1
		else:
			temp_label_list[i] = 1

	print "-------------------"+ print_name +"--------------------"
	for i in temp_label_list:
		print i + ":" + str(temp_label_list[i])
	print "\n"

def build_knn_raw_pixels(trainRI, testRI, trainRL, testRL):
	# train and evaluate a k-NN classifer on the raw pixel intensities
	#print("[INFO] evaluating raw pixel accuracy...")
	model = KNeighborsClassifier(n_neighbors=args["neighbors"],
		n_jobs=args["jobs"])
	model.fit(trainRI, trainRL)
	acc = model.score(testRI, testRL)
	#print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
	return (acc*100)


def build_knn_histogram(trainFeat, testFeat, trainLabels, testLabels):
	# train and evaluate a k-NN classifer on the histogram
	# representations
	#print("[INFO] evaluating histogram accuracy...")
	model = KNeighborsClassifier(n_neighbors=args["neighbors"],
		n_jobs=args["jobs"])
	model.fit(trainFeat, trainLabels)
	acc = model.score(testFeat, testLabels)
	#print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
	return (acc*100)

def join_clusters(f_index_to_cluster, f_cluster_list, f_cluster_count, p, q):
	if(len(f_cluster_list[q]) + len(f_cluster_list[p])) > 6 :
		return (f_index_to_cluster, f_cluster_list)	
	for i in f_cluster_list[q]:
		f_index_to_cluster[i] = p
		f_cluster_list[p].append(i)

	for i in range(q, f_cluster_count-1):
		f_cluster_list[i] = f_cluster_list[i+1]
	
	return (f_index_to_cluster, f_cluster_list)

def build_cluster(imagePaths):
	result_temp_index_to_cluster = index_to_cluster
	result_temp_cluster_list = cluster_list
	result_acc = 0

	for p in range(0,max_clusters):
		for q in range(p+1,max_clusters):
			print "combining cluster " + str(p) + " and " + str(q)
			# initialize the raw pixel intensities matrix, the features matrix,
			# and labels list
			rawImages = []
			features = []
			labels = []
			temp_index_to_cluster = index_to_cluster[:]
			temp_cluster_list = []
			for i in cluster_list:
				t = []
				for j in i:
					t.append(j)
				temp_cluster_list.append(t)

			(temp_index_to_cluster, temp_cluster_list) = join_clusters(temp_index_to_cluster, temp_cluster_list, max_clusters, p, q)		

			# loop over the input images
			for (i, imagePath) in enumerate(imagePaths):
				# load the image and extract the class label (assuming that our
				# path as the format: /path/to/dataset/{class}/{image_num}.jpg
				image = cv2.imread(imagePath)
				label = str(temp_index_to_cluster[class_list.index(imagePath.split("/")[-2])])

				# extract raw pixel intensity "features", followed by a color
				# histogram to characterize the color distribution of the pixels
				# in the image
				pixels = extract_hog_histogram(image)
				hist = extract_color_histogram(image)

				# update the raw images, features, and labels matricies,
				# respectively
				rawImages.append(pixels)
				features.append(hist)
				labels.append(label)

				# show an update every 1,000 images
				#if i > 0 and i % 50 == 0:
					#print("[INFO] processed {}/{}".format(i, len(imagePaths)))

			# show some information on the memory consumed by the raw images
			# matrix and features matrix
			rawImages = np.array(rawImages)
			features = np.array(features)
			labels = np.array(labels)
			'''
			print("[INFO] pixels matrix: {:.2f}MB".format(
				rawImages.nbytes / (1024 * 1000.0)))
			print("[INFO] features matrix: {:.2f}MB".format(
				features.nbytes / (1024 * 1000.0)))
			'''

			# partition the data into training and testing splits, using 75%
			# of the data for training and the remaining 25% for testing
			(trainRI, testRI, trainRL, testRL) = train_test_split(
				rawImages, labels, test_size=0.25, random_state=42)
			(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
				features, labels, test_size=0.25, random_state=42)

			#print_count_each_class("train class count", trainRL)
			#print_count_each_class("text class count", testRL)

			acc = build_knn_raw_pixels(trainRI, testRI, trainRL, testRL)
			print "accuracy : " + str(acc)
			if acc > result_acc:
				result_acc = acc
				result_temp_index_to_cluster = temp_index_to_cluster
				result_temp_cluster_list = temp_cluster_list

	return (result_acc, result_temp_index_to_cluster, result_temp_cluster_list)

def check_for_cluster(imagePaths):
	# initialize the raw pixel intensities matrix, the features matrix,
	# and labels list
	rawImages = []
	features = []
	labels = []

	# loop over the input images
	for (i, imagePath) in enumerate(imagePaths):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}/{image_num}.jpg
		image = cv2.imread(imagePath)
		label = str(check_index_to_cluster[class_list.index(imagePath.split("/")[-2])])

		# extract raw pixel intensity "features", followed by a color
		# histogram to characterize the color distribution of the pixels
		# in the image
		#pixels = image_to_feature_vector(image)		
		pixels = extract_hog_histogram(image)
		hist = extract_hog_histogram(image)

		# update the raw images, features, and labels matricies,
		# respectively
		rawImages.append(pixels)
		features.append(hist)
		labels.append(label)

		# show an update every 1,000 images
		#if i > 0 and i % 50 == 0:
			#print("[INFO] processed {}/{}".format(i, len(imagePaths)))

	# show some information on the memory consumed by the raw images
	# matrix and features matrix	
	rawImages = np.array(rawImages)
	features = np.array(features)
	labels = np.array(labels)
	'''
	print("[INFO] pixels matrix: {:.2f}MB".format(
		rawImages.nbytes / (1024 * 1000.0)))
	print("[INFO] features matrix: {:.2f}MB".format(
		features.nbytes / (1024 * 1000.0)))
	'''

	# partition the data into training and testing splits, using 75%
	# of the data for training and the remaining 25% for testing
	(trainRI, testRI, trainRL, testRL) = train_test_split(
		rawImages, labels, test_size=0.20, random_state=42)
	(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
		features, labels, test_size=0.25, random_state=42)

	print_count_each_class("train class count", trainRL)
	print_count_each_class("test class count", testRL)

	acc = build_knn_raw_pixels(trainRI, testRI, trainRL, testRL)
	#acc = build_knn_raw_pixels(trainRI, trainRI, trainRL, trainRL)
	print "accuracy : " + str(acc)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
#print("[INFO] describing images...")
imagePaths = []

for i in class_list:
	imagePaths += [(args["dataset"] + "/" + i + "/" + f) for f in listdir(args["dataset"] + "/" + i)]


check_for_cluster(imagePaths)

'''
final_accuracy = 0
for i in range(0,20):
	if(max_clusters < 4):
		break

	(round_accuracy, temp_index_to_cluster, temp_cluster_list) = build_cluster(imagePaths)
	
	if(round_accuracy < final_accuracy):
		break
	else:
		final_accuracy = round_accuracy
		max_clusters -= 1
		index_to_cluster = temp_index_to_cluster
		cluster_list = temp_cluster_list

	print "\n-------------------------------round " + str(i) + "------------------------------"
	print final_accuracy
	print str(index_to_cluster)
	print str(cluster_list)
	print "---------------------------------------------------------------------\n"
'''
