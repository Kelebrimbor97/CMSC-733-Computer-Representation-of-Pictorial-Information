#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Nishad Kulkarni (nkulkar2@umd.edu; nishadkulkarni97@gmail.com)
Professional M.Eng in Robotics (PMRO)
Spring 2022
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# Add any python libraries here


"""
Function to read images from given path and number of images to be read.
Function assumes that images are numbered as '1.jpg','2.jpg', ... ,'img_nums.jpg'
"""

def image_set_reader(path,img_nums):

	im_set_gray = []
	im_set_color = []

	for i in range(img_nums):
		imtmp = cv2.imread(path+str(i+1)+".jpg")
		imtmp = cv2.cvtColor(imtmp,cv2.COLOR_BGR2RGB)
		im_set_color.append(imtmp)
		img = cv2.cvtColor(imtmp,cv2.COLOR_BGR2GRAY)
		#print("Gray image shape:",np.shape(img))
		im_set_gray.append(img)
	
	return im_set_gray,im_set_color

"""
Detect corners using cv2.goodFeaturesToTrack and return corners and images with corners to display
"""

def corner_detection(path,imnums):

	img_set_gray,img_set_color = image_set_reader(path,imnums)

	corner_img_set =[]
	corners =[]

	for i in range(imnums):

		corner_img = cv2.goodFeaturesToTrack(img_set_gray[i],10000,0.01,10)
		corner_img = np.int0(corner_img)
		corners.append(corner_img)

		#Here corner_img stores locations of detected corners

		for j in corner_img:
			x,y = j.ravel()
			cv2.circle(img_set_color[i],(x,y),3,(255,162,0),-1)

		
		corner_img_set.append(img_set_color[i])

	return corners, corner_img_set
"""
Get anms corners sets
"""

def anms(path,imnums,corners,Nbest):

	anms_corners = []

	img_set_gray,_ = image_set_reader(path,imnums)

	for i in range(imnums):
		
		current_corners = corners[i]
		current_gray_img = img_set_gray[i]
		# current_color_img = og_img_set_color[i]
		# current_cornered = corners[i]
		# copy_for_anms = img_set_color[i]
		l = len(current_corners)
		ri = np.multiply(np.ones(l),np.inf)
		ED=0

		for j in range(l):

			xj,yj = current_corners[j].ravel()

			for k in range(l):

				xk,yk = current_corners[k].ravel()

				if (current_gray_img[yj][xj]>current_gray_img[yk][xk]):

					ED = np.square(xj - xk) + np.square(yj - yk)

				if ED<ri[j]:

					ri[j] = ED

		indexes = np.argsort(ri)
		indexes = indexes[-Nbest:]
		print("Indexes len:",len(indexes),";Current corners len:",np.shape(current_corners))
		tmp_anms =[]
		for n in range(Nbest):
			tmp_anms.append(current_corners[indexes[n]])
		print("Current anms corner size:",np.shape(tmp_anms))
		anms_corners.append(tmp_anms)

	return anms_corners

"""
Display Corners/Features detected in normal corner detection and ANMS
"""

def corner_displayer(path,imnums,corner_imgs,anms):

	_, img_set_color = image_set_reader(path,imnums)
	_,anms_set = image_set_reader(path,imnums)

	for i in range(imnums):
		current_color_img = img_set_color[i]
		current_cornered = corner_imgs[i]
		copy_for_anms = anms_set[i]
		tmp_anms = anms[i]

		for j in tmp_anms:
			#print(np.shape(j))
			x,y = j.ravel()
			cv2.circle(copy_for_anms,(x,y),3,(0,255,255),-1)

		plt.figure()
		plt.subplot(1,3,1)
		plt.axis("off")
		plt.imshow(current_color_img)

		plt.subplot(1,3,2)
		plt.axis("off")
		plt.imshow(current_cornered)

		plt.subplot(1,3,3)
		plt.axis("off")
		plt.imshow(copy_for_anms)

		plt.tight_layout()
	plt.show()

"""
Calculate Nbest as 64% of mean number of carner points 
"""

def nbest_selector(corners):

	l = []
	for i in corners: l.append(len(i))
	l = int(np.multiply(np.mean(l),0.64))

	return l

"""
Extract Feature descriptors such that features that are 20 pixels within the borders are ignored.
It the returns corresponding anms points in anms that have been used for finding these features
"""

def FeatureDescriptor(path,imnums,anms):

	img_set_gray, _ = image_set_reader(path,imnums)

	patch_set_per_image = []
	anms_tapered = []	

	for i in range(imnums):
		
		current_gray = img_set_gray[i]
		current_anms = anms[i]
		a,b = np.shape(current_gray)
		current_patch_set = []
		current_tapered_anms =[]

		#plt.figure(i+1)
		counter = 0

		for j in current_anms:

			y,x = j.ravel()
			
			start_x = x-19
			end_x = x+21

			start_y = y-19
			end_y = y+21

			if((end_x>a) or (start_x<0) or (end_y>b) or (start_y<0)):
				continue

			
			counter+= 1

			#plt.subplot(int(len(current_anms)/10),10,counter)
			#plt.axis("off")
			
			patch = current_gray[int(start_x):int(end_x),int(start_y):int(end_y)]
			gaussed_patch = cv2.GaussianBlur(patch,(5,5),cv2.BORDER_DEFAULT)

			subsample = gaussed_patch[16:24,16:24]
			subsample = np.reshape(subsample,64)
			subsample = np.subtract(subsample,np.mean(subsample))
			subsample = np.divide(subsample,np.std(subsample))
			current_patch_set.append(subsample)
			current_tapered_anms.append(j)

			#subsample_image = np.reshape(subsample,(8,8))
			#plt.imshow(subsample_image,cmap="gray")

		patch_set_per_image.append(current_patch_set)
		anms_tapered.append(current_tapered_anms)
		#plt.tight_layout()
		#plt.savefig("Image_"+str(i+1)+"_patches.png")

	#plt.show()

	return patch_set_per_image,anms_tapered

"""
Match features using two images at a time, display the matches, and return selected anms corners for each image
"""

def FeatureMatching_two_imgs(imnum1,imnum2,features_set,anms_tapered,path,imnums):

	print("Feature Matching started...")
	_,img_set_color = image_set_reader(path, imnums)
	features_img1 = features_set[imnum1]
	features_img2 = features_set[imnum2]
	anms_1 = anms_tapered[imnum1]
	anms_2 = anms_tapered[imnum2]
	match_set = []

	for i in range(len(features_img1)):

		ed_set = []
		match_pair = []

		for j in range(len(features_img2)):

			ed = np.sum(np.square(np.subtract(features_img1[i],features_img2[j])))
			ed_set.append(ed)

		m0 = np.argsort(ed_set)[0]
		m1 = np.argsort(ed_set)[1]

		if((ed_set[m0]/ed_set[m1])<0.64):
			match_pair = [i,m0]
			match_set.append(match_pair)
	"""
	Display matches beyond this point
	"""

	a = np.max([np.shape(img_set_color[imnum1])[0],np.shape(img_set_color[imnum2])[0]])
	b1 = np.shape(img_set_color[imnum1])[1]
	#print("b1=",b1)
	b2 = np.shape(img_set_color[imnum2])[1]
	#print("b2=",b2)
	b = b1+b2
	c = np.max([np.shape(img_set_color[imnum1])[2],np.shape(img_set_color[imnum2])[2]])

	matchimg_base = np.zeros((a,b,c),type(img_set_color[imnum1].flat[0]))
	matchimg_base[:,:b1] = img_set_color[imnum1]
	matchimg_base[:,b1:] = img_set_color[imnum2]

	matchpair_x,matchpair_y = np.shape(match_set)
	print("Match set size:",matchpair_x,",",matchpair_y)
	for i in range(np.shape(anms_1)[0]):
		c1 = anms_1[i][0]
		cv2.circle(matchimg_base,c1,3,255)
		for j in range(np.shape(anms_2)[0]):
			x= anms_2[j][0][0]+b1
			y = anms_2[j][0][1]
			cv2.circle(matchimg_base,(x,y),3,(0,255,0))

	for i in match_set:
		u,v = i
		x1,y1 = anms_1[u][0]
		x2 = anms_2[v][0][0]+b1
		y2 = anms_2[v][0][1]
		cv2.line(matchimg_base,(x1,y1),(x2,y2),(0,0,255))

	titlestr = "Feature Matching "+str(imnum1+1)+" to "+str(imnum2+1)
	plt.figure(titlestr)
	plt.axis("off")
	plt.imshow(matchimg_base)
	plt.show()	

	return match_set

"""
RANSAC!
"""

def ransac(m,anms_tapered,t):

	Nmax = 2500
	f1 =[]
	f2 =[]

	match_set = m[0]
	imnum1 = m[1]
	imnum2 = m[2]

	#print("RANSAC initiated for images",imnum1,"and",imnum2)

	anms1 = anms_tapered[imnum1]
	anms2 = anms_tapered[imnum2]
	total_inliers = 0
	Hf = []

	for i in range(Nmax):

		indexes = []
		rind = np.random.randint(len(match_set),size=4)

		p1 = []
		p2 = []

		for j in rind: p1.append(anms1[match_set[j][0]][0])
		for j in rind: p2.append(anms2[match_set[j][1]][0])

		p1 = np.array(p1,np.float32)
		p2 = np.array(p2,np.float32)

		H = cv2.getPerspectiveTransform(p1,p2)

		#print("perspective transform:",np.shape(H))

		inliers = 0

		for k in range(len(match_set)):

			u,v = match_set[k]

			src = np.array([anms1[u][0][1],anms1[u][0][0],1])
			dst = np.array([anms2[v][0][1],anms2[v][0][0]])

			est = np.matmul(H,src)

			if(est[2]==0):
				est[2]=0.0000001

			est_x = est[0]/est[2]
			est_y = est[1]/est[2]

			estd = np.array([est_x,est_y])

			ed = np.sqrt(np.sum(np.square(np.subtract(dst,estd))))
			
			if(ed)<t:
				inliers+=1
				indexes.append(k)

		s1 =[]
		s2 =[]

		if total_inliers<inliers:
			total_inliers=inliers
			print("Total inliers:",total_inliers)
			for y in indexes:
				u,v = match_set[y] 
				s1.append(anms1[u][0])
				s2.append(anms2[v][0])

			f1 = s1
			f2 = s2

	Hf,_ = cv2.findHomography(np.array(f1),np.array(f2))
	
	return f1,f2,Hf

"""
match display
"""
def match_displayer(s1,s2,imnum1,imnum2,path,imgnums):

	print("S1 size:",np.shape(s1))
	print("S2 size:",np.shape(s2))

	_,img_set_color = image_set_reader(path,imgnums)

	a = np.max([np.shape(img_set_color[imnum1])[0],np.shape(img_set_color[imnum2])[0]])
	b1 = np.shape(img_set_color[imnum1])[1]
	#print("b1=",b1)
	b2 = np.shape(img_set_color[imnum2])[1]
	#print("b2=",b2)
	b = b1+b2
	c = np.max([np.shape(img_set_color[imnum1])[2],np.shape(img_set_color[imnum2])[2]])

	matchimg_base = np.zeros((a,b,c),type(img_set_color[imnum1].flat[0]))
	matchimg_base[:,:b1] = img_set_color[imnum1]
	matchimg_base[:,b1:] = img_set_color[imnum2]

	for i in range(len(s1)):

		x1 = s1[i][0]
		y1 = s1[i][1]

		x2 = s2[i][0]+b1
		y2 = s2[i][1]

		cv2.circle(matchimg_base,(x1,y1),3,255)
		cv2.circle(matchimg_base,(x2,y2),3,(0,0,255))

		cv2.line(matchimg_base,(x1,y1),(x2,y2),(0,255,0))

	titlestr = "RANSAC_"+str(imnum1+1)+"_to_"+str(imnum2+1)
	plt.figure(titlestr)
	plt.axis("off")
	plt.imshow(matchimg_base)
	plt.show()

def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""

	path = os.getcwd()+"/Data/Train/Set3/"
	img_nums = 8

	_,img_set_color = image_set_reader(path,img_nums)

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	corners_set, corner_img_set = corner_detection(path,img_nums)

	#Here corners x and y are inverted
	
	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	Nbest = nbest_selector(corners_set)
	print("Nbest points for this set of images:",Nbest)
	anms_corners = anms(path,img_nums,corners_set,Nbest)

	corner_displayer(path,img_nums,corner_img_set,anms_corners)


	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	
	features_set,tapered_anms = FeatureDescriptor(path,img_nums,anms_corners)
	
	for j in range(len(features_set)):
		print("Size of features set for image",j+1,":", np.shape(features_set[j]))
		print("Size of anms set for image",j+1,":", np.shape(tapered_anms[j]))


	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""
	matching_set = {}
	for i in range(img_nums):

		for j in range(img_nums):

			if(j==i):continue

			key_str = "matchset_"+str(i+1)+"_to_"+str(j+1)
			print(key_str)
			m = FeatureMatching_two_imgs(i,j,features_set,tapered_anms,path,img_nums)
			matching_set[key_str] = [m,i,j]

	# match01 = FeatureMatching_two_imgs(0,1,features_set,tapered_anms,path,img_nums)
	# match12 = FeatureMatching_two_imgs(1,2,features_set,tapered_anms,path,img_nums)
	# match20 = FeatureMatching_two_imgs(2,0,features_set,tapered_anms,path,img_nums)

	"""
	Refine: RANSAC, Estimate Homography
	"""

	# k1 = "matchset_2_to_1"
	# k2 = "matchset_3_to_2"
	# s1_2_1,s1_2_2,H1_2 = ransac(matching_set[k1],tapered_anms,t=20)
	# s2_3_1,s2_3_2,H2_3 = ransac(matching_set[k2],tapered_anms,t=20)
	# match_displayer(s1_2_1,s1_2_2,matching_set[k1][1],matching_set[k1][2],path,img_nums)
	# match_displayer(s2_3_1,s2_3_2,matching_set[k2][1],matching_set[k2][2],path,img_nums)

	Homographies = {}
	warp_images = {}

	for i in range(img_nums):

		for j in range(img_nums):
			if(j==i or ((i-j)>1) or((j-i)>1)):continue
			k = "matchset_"+str(i+1)+"_to_"+str(j+1)
			s1,s2,h = ransac(matching_set[k],tapered_anms,t=32)
			match_displayer(s1,s2,matching_set[k][1],matching_set[k][2],path,img_nums)

			Homographies[k] = h
			warps = cv2.warpPerspective(img_set_color[i],h,np.shape(img_set_color[j])[0:2])
			warp_images[k] = warps
			print("Shape of Warp:",np.shape(warps),"and type",type(warps))

	print(len(Homographies))

	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	for i in warp_images:

		print(i)
		# plt.imshow(i)
		# plt.show()

if __name__ == '__main__':
    main()