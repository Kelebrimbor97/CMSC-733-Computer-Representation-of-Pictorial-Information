#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

from cv2 import COLOR_BGRA2RGB
import numpy as np
import cv2
import matplotlib.pyplot as mpy
import scipy.signal as con
import scipy.ndimage as rot
import os

from sklearn.cluster import KMeans



def gauss_matrix(sigx,sigy,l):
	mat = np.zeros((l,l))
	mu = int(l/2)
	for i in range(l):
		for j in range(l):
			mat[i,j] = (np.exp(-(np.square(i-mu))/(2*np.square(sigx))))*(np.exp(-(np.square(j-mu))/(2*np.square(sigy))))/(2*np.pi*sigx*sigy)
			#/(np.square(sig)*(2*np.pi))
	return mat
#function for 9x9 sobel filter
def sobel():
	return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])



def dog_filters():
	sob = sobel()
	gauss_small = gauss_matrix(3,3,45)
	dog_filter_basis_small = con.convolve2d(gauss_small,sob)
	gauss_large = gauss_matrix(7.5,7.5,45)
	dog_filter_basis_large = con.convolve2d(gauss_large,sob)
	dog_filter_bank=[]
	for n in range(16):
		f = rot.rotate(dog_filter_basis_small,(n*22.5),reshape=False)
		dog_filter_bank.append(f)
	for n in range(16):
		f = rot.rotate(dog_filter_basis_large,(n*22.5),reshape=False)
		dog_filter_bank.append(f)
	"""
	for m in range(2):
		for n in range(32):
			mpy.subplot(2,16,n+1)
			mpy.axis("off")
			mpy.tight_layout()
			mpy.imshow(dog_filter_bank[n],cmap="gray")
	mpy.show()
	"""
	return dog_filter_bank



def log_filter(sig,l):
	mu = int(l/2)
	mat = np.zeros((l,l))
	
	for i in range(l):
		for j in range(l):
			mat[i,j] = np.array(((np.square(i-mu)+np.square(j-mu))/(2*np.square(sig)))-1)*np.exp(-(np.square(i-mu)+np.square(j-mu))/(2*np.square(sig)))
			
	return mat



def leung_malik():
	
	root_two = float(np.sqrt(2))
	lml_sigmax = np.array([1,root_two,2,(2*root_two)])
	lml_sigmay = np.multiply(3,lml_sigmax)
	lm = []
	sob = sobel()
	for i in range(3):
		mat = gauss_matrix(lml_sigmax[i+1],lml_sigmay[i+1],45)
		holder1 = con.convolve2d(mat,sob)
		for j in range(6):
			rotator1 = rot.rotate(holder1,j*-30)
			lm.append(rotator1)
		holder2 = con.convolve2d(holder1,sob)
		for j in range(6):
			rotator2 = rot.rotate(holder2,j*-30)
			lm.append(rotator2)
	for i in range(4):
		mat = log_filter(lml_sigmax[i],45)
		lm.append(mat)
	
	for i in range(4):
		mat = log_filter(lml_sigmay[i],45)
		lm.append(mat)
	
	for i in range(4):
		mat = gauss_matrix(lml_sigmax[i],lml_sigmax[i],45)
		lm.append(mat)
	"""
	for i in range(len(lm)):
		
		#save_str = "filt" + str(i+1) + ".png"
		mpy.subplot(4,12,i+1)
		mpy.axis("off")
		mpy.imshow(lm[i],cmap="gray")
		#mpy.savefig(save_str)
	mpy.tight_layout()
	mpy.show()
	"""

	return lm
	



def Gabor_filtering(sigma,lam,theta):
	
	l = 271
	mu = int(l/2)
	mat = np.zeros((l,l))
	
	for i in range(l):
		for j in range(l):
			xd = (i-mu)*np.cos(theta) + (j-mu)*np.sin(theta)
			yd = -(i-mu)*np.sin(theta) + (j-mu)*np.cos(theta)
			mat[i,j] = np.exp(-(np.square(xd)+np.square(yd))/(2*np.square(sigma))) * np.cos(2*np.pi*xd/lam)
	
	return mat



def gb_filter_bank(sigma,lam,theta):
	
	gbf = []
	d = 22.5
	for i in range(5):
		s = sigma + (i*10)
		l = lam + (i*5)
		gabor = Gabor_filtering(s,l,theta)
		for r in range(8):
			hold = rot.rotate(gabor,(r*d))
			gbf.append(hold)
	
	return gbf



def half_disk_filter_bank():
	s = [16,24,32,40]
	hdf = []
	d = 180/8
	for i in range(len(s)):
		l = int(s[i])
		m = np.zeros((l,l))
		mu = int(s[i]/2)
		for j in range(l):
			for k in range(l):
				r = np.square(mu) - np.square(j-mu+0.5) - np.square(k-mu+0.5)
				if  ((r>0) and k<mu):
					m[j,k] = 1
		for x in range(16):
			a = rot.rotate(m,(x*d),reshape=False)
			am = np.mean(a)
			a = a+am
			a = np.floor(a)
			hdf.append(a)
			b = rot.rotate(a,180)
			bm = np.mean(b)
			b = b+am
			b = np.floor(b)
			hdf.append(b)

	# for i in range(len(hdf)):
	# 	mpy.subplot(6,8,i+1)
	# 	mpy.axis("off")
	# 	mpy.imshow(hdf[i],cmap="gray")
	# mpy.show()

	return hdf



def Texton_map(imnum,dog):
	dog_len = len(dog)

	#The following statement assumes that the current working directory is that of 'Phase 1' with the child directory being BSDS500

	imtmp = cv2.imread(os.getcwd()+"/BSDS500/Images/"+str(imnum)+".jpg")
	#print(imtmp)
	img = cv2.cvtColor(imtmp,cv2.COLOR_BGR2GRAY)
	texton = np.zeros((img.size,len(dog)))

	for i in range(dog_len):
		temp_img = cv2.filter2D(img,-1,dog[i])
		temp_img = temp_img.reshape((1,temp_img.size))
		texton[:,i] = temp_img
	kmeans = KMeans(n_clusters=256,n_init=4)
	kmeans.fit(texton)
	labels = kmeans.labels_
	texton = np.reshape(labels,(img.shape))

	# mpy.axis("off")
	# mpy.imshow(texton)
	# mpy.show()

	return texton



def Brightness_map(imnum):

	imtmp = cv2.imread(os.getcwd()+"/BSDS500/Images/"+str(imnum)+".jpg")
	img = cv2.cvtColor(imtmp,cv2.COLOR_BGR2GRAY)
	b = img.reshape((img.shape[0]*img.shape[1]),1)
	kmeans = KMeans(n_clusters=256)
	kmeans.fit(b)
	labels = kmeans.labels_
	bright = np.reshape(labels,(img.shape[0],img.shape[1]))
	brightmap = 255*(bright-np.min(bright))/float((np.max(bright)-np.min(bright)))

	return brightmap

def Color_map(imnum):

	imtmp = cv2.imread(os.getcwd()+"/BSDS500/Images/"+str(imnum)+".jpg")
	b = imtmp.reshape((imtmp.shape[0]*imtmp.shape[1]),3)
	kmeans = KMeans(n_clusters=256)
	kmeans.fit(b)
	labels = kmeans.labels_
	bright = np.reshape(labels,(imtmp.shape[0],imtmp.shape[1]))
	cmap = 255*(bright-np.min(bright))/float((np.max(bright)-np.min(bright)))

	return cmap

def Gradient_Maker(img,numbins,hdf):
    grad = img
    for i in range(int(len(hdf)/2)):
        g = chi_dist(img,numbins,hdf[2*i],hdf[2*i+1])
        grad = np.dstack((grad,g))
    mean = np.mean(grad,axis=2)
    return mean

def chi_dist(img,numbins,hdf_l,hdf_r):
    chi_sq_dist = img*0
    g =[]
    h =[]
    for i in range(numbins):
        #imgtemp = np.ma.masked_where(img == i,img)
        #imgtemp = img.mask.astype(np.int)
        g = cv2.filter2D(img,-1,hdf_l)
        h = cv2.filter2D(img,-1,hdf_r)
        chi_sq_dist = chi_sq_dist + ((g-h))**2/(g+h+0.0001)
    return chi_sq_dist

def main():

	figure_counter = 1
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""

	dog = dog_filters()

	mpy.figure(figure_counter)
	for i in range(int(len(dog))):
		mpy.subplot(2,int(len(dog)/2),i+1)
		mpy.axis("off")
		mpy.imshow(dog[i],cmap="gray")
	mpy.tight_layout()
	mpy.savefig("DoG.png")
	mpy.show()

	figure_counter+=figure_counter

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	lm = leung_malik()
	mpy.figure(figure_counter)
	for i in range(int(len(lm))):
		mpy.subplot(4,int(len(lm)/4),i+1)
		mpy.axis("off")
		mpy.imshow(lm[i],cmap="gray")
	mpy.tight_layout()
	mpy.savefig("LM.png")
	mpy.show()
	
	figure_counter+=figure_counter

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	gbf = gb_filter_bank(15,40,0)

	mpy.figure(figure_counter)
	for i in range(int(len(gbf))):
		mpy.subplot(5,int(len(gbf)/5),i+1)
		mpy.axis("off")
		mpy.imshow(gbf[i],cmap="gray")
	mpy.tight_layout()
	mpy.savefig("Gabor.png")
	mpy.show()
	
	figure_counter+=figure_counter
	
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""

	hdf = half_disk_filter_bank()

	mpy.figure(figure_counter)
	for i in range(int(len(hdf))):
		mpy.subplot(8,int(len(hdf)/8),i+1)
		mpy.axis("off")
		mpy.imshow(hdf[i],cmap="gray")
	mpy.tight_layout()
	mpy.savefig("HDMasks.png")
	mpy.show()
	
	figure_counter+=figure_counter

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	
	pblite = []

	for i in range(10):

		figure_counter+=figure_counter

		sobeline = np.multiply(cv2.imread(os.getcwd()+"/BSDS500/SobelBaseline/"+str(i+1)+".png",0),0.5)
		cannyline = np.multiply(cv2.imread(os.getcwd()+"/BSDS500/CannyBaseline/"+str(i+1)+".png",0),0.5)

		print("\n\nStarting Processing of Image ",i+1)

		tmaps = Texton_map(i+1,dog+lm)
		bmaps = Brightness_map(i+1)
		cmaps = Color_map(i+1)


		print("T,B,C done for image ",(i+1))

		mpy.figure(figure_counter+i+1)
		mpy.subplot(2,3,1)
		mpy.axis("off")
		mpy.imshow(tmaps)
		mpy.subplot(2,3,2)
		mpy.axis("off")
		mpy.imshow(bmaps)
		mpy.subplot(2,3,3)
		mpy.axis("off")
		mpy.imshow(cmaps)

		tgmaps = Gradient_Maker(tmaps,16,hdf)
		print("Tgmaps done")
		bgmaps = Gradient_Maker(bmaps,16,hdf)
		print("Bgmaps done")
		cgmaps = Gradient_Maker(cmaps,16,hdf)
		print("Cgmaps done")

		mpy.subplot(2,3,4)
		mpy.axis("off")
		mpy.imshow(tgmaps)
		mpy.subplot(2,3,5)
		mpy.axis("off")
		mpy.imshow(bgmaps)
		mpy.subplot(2,3,6)
		mpy.axis("off")
		mpy.imshow(cgmaps)

		print("Done with image",(i+1))


		tg = np.divide(np.add(tgmaps,np.add(bgmaps,cgmaps)),3)

		pbl = np.multiply(tg,(np.multiply(sobeline,cannyline)))
		save_str = "PB_Lite_OP"+str(i+1)+".png"
		cv2.imwrite(save_str,pbl)

		pblite.append(pbl)
		
	
	mpy.show()
	
	for i in range(10):

		figure_counter+=figure_counter
		mpy.figure(figure_counter+i+1)
		mpy.axis("off")
		mpy.imshow(pblite[i])

	mpy.show()

	
	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    
if __name__ == '__main__':
    main()
 


