# Homework 0 - Shake My Boundary

Edge detection is one of the most important steps in CV. In this project we examine the various methods of Edge Detection on a number of pictures. Further, we also examine the combinations of these edge detection methods.

![og_imgs](https://user-images.githubusercontent.com/35636842/218597493-6969f533-1134-48a9-a085-1deaf7ce89ef.gif)![baseline](https://user-images.githubusercontent.com/35636842/218597491-6745a414-5fd5-4c87-879c-1431f5b60c81.gif)

---

## A. Introduction

In this project we analyse the use of PB(Probability of Boundary) method to detect edges in an image. We measure them up against Canny and Sobel edges. For implementing PB method we use combination of 3 types of filter banks:

### 1. Oriented Derivative of Gaussian(DoG) filters

A simple but effective filter bank is a collection of oriented Derivative of Gaussian (DoG) filters. These filters can be created by convolving a simple Sobel filter and a Gaussian kernel and then rotating the result. Suppose we want o orientations (from 0 to 360∘) and s scales, we should end up with a total of s×o filters. A sample filter bank of size 2×16 with 2 scales and 16 orientations is shown below.

![Dog_Filter_Bank](/Homework%200%20-%20Shake%20My%20Boundary/Phase1/Code/Resulting_Images_Extra/DoG.png)

### 2. Leung - Malik Filters

The Leung-Malik filters or LM filters are a set of multi scale, multi orientation filter bank with 48 filters. It consists of first and second order derivatives of Gaussians at 6 orientations and 3 scales making a total of 36; 8 Laplacian of Gaussian (LOG) filters; and 4 Gaussians. We consider two versions of the LM filter bank. In LM Small (LMS), the filters occur at basic scales σ={1,2–√,2,22–√}. The first and second derivative filters occur at the first three scales with an elongation factor of 3, i.e., (σx=σ and σy=3σx). The Gaussians occur at the four basic scales while the 8 LOG filters occur at σ and 3σ. For LM Large (LML), the filters occur at the basic scales σ={2–√,2,22–√,4}.

![Leung_Malik_Filter_Bank](/Homework%200%20-%20Shake%20My%20Boundary/Phase1/Code/Resulting_Images_Extra/Leung_Malik.png)

### 3. Gabor Filters

Gabor Filters are designed based on the filters in the human visual system. A gabor filter is a gaussian kernel function modulated by a sinusoidal plane wave.

![Gabor_Filter_Bank](/Homework%200%20-%20Shake%20My%20Boundary/Phase1/Code/Resulting_Images_Extra/Gabor.png)

--- 

## B. Texton Map - T

Filtering an input image with each element of the filter bank results in a vector of fillter responses centered on each pixel. For instance, if the filter bank has N filters, you’ll have N filter responses at each pixel. A distribution of these N-dimensional filter responses could be thought of as encoding texture properties. We will simplify this representation by replacing each N-dimensional vector with a discrete texton ID. We will do this by clustering the filter responses at all pixels in the image in to K textons using kmeans. Each pixel is then represented by a one dimensional, discrete cluster ID instead of a vector of high-dimensional, real-valued filter responses (this process of dimensionality reduction from N to 1 is called “Vector Quantization”). This can be represented with a single channel image with values in the range of [1,2,3,⋯,K].

---

## C. Brightness Map - B

The concept of the brightness map is as simple as capturing the brightness changes in the image. Here, again we cluster the brightness values using kmeans clustering (grayscale equivalent of the color image) into a chosen number of clusters. We call the clustered output as the brightness map B.

---

## D. Color Map - C

The concept of the color map is to capture the color changes or chrominance content in the image. Here, again we cluster the color values using kmeans clustering into a chosen number of clusters. We call the clustered output as the color map C.

---

## E. Texture, Brightness and Color Gradients Tg,Bg,Cg

To obtain Tg,Bg,Cg, we need to compute differences of values across different shapes and sizes. This can be achieved very efficiently by the use of Half-disc masks.

![Half_Disk_Maps](/Homework%200%20-%20Shake%20My%20Boundary/Phase1/Code/Resulting_Images_Extra/HDMasks.png)

The half-disc masks are simply (pairs of) binary images of half-discs. This is very important because it will allow us to compute the χ2 (chi-square) distances (finally obtain values of Tg,Bg,Cg) using a filtering operation, which is much faster than looping over each pixel neighborhood and aggregating counts for histograms. Forming these masks is quite trivial.

Tg,Bg,Cg encode how much the texture, brightness and color distributions are changing at a pixel. We compute Tg,Bg,Cg

by comparing the distributions in left/right half-disc pairs centered at a pixel. If the distributions are the similar, the gradient should be small. If the distributions are dissimilar, the gradient should be large. Because our half-discs span multiple scales and orientations, we will end up with a series of local gradient measurements encoding how quickly the texture or brightness distributions are changing at different scales and angles.

We will compare texton, brightness and color distributions with the χ2
measure. The χ2 distance is a frequently used metric for comparing two histograms. χ2 distance between two histograms g and h

with the same binning scheme is defined as follows:

χ2(g,h)=12∑i=1K(gi−hi)2gi+hi

here, K

indexes though the bins. Note that the numerator of this expression is simply the sum of squared difference between histogram elements. The denominator adds a “soft” normalization to each bin so that less frequent elements still contribute to the overall distance.

To effciently compute Tg,Bg,Cg, filtering can used to avoid nested loops over pixels. In addition, the linear nature of the formula above can be exploited. At a single orientation and scale, we can use a particular pair of masks to aggregate the counts in a histogram via a filtering operation, and compute the χ2 distance (gradient) in one loop over the bins according to the following outline:
```
chi_sqr_dist = img*0

for i = 1:num_bins
	tmp = 1 where img is in bin i and 0 elsewhere
	g_i = convolve tmp with left_mask
	h_i = convolve tmp with right_mask
	update chi_sqr_dist
end
```

The above procedure should generate a 2D matrix of gradient values. Simply repeat this for all orientations and scales, you should end up with a 3D matrix of size m×n×N, where (m,n) are dimensions of the image and N is the number of filters.

---

## F. Pb-Lite Output

The final step is to combine information from the features with a baseline method (based on Sobel or Canny edge detection or an average of both) using a simple equation
PbEdges=(Tg+Bg+Cg)3⊙(w1∗cannyPb+w2∗sobelPb)

Here, ⊙
is the Hadamard product operator. A simple choice for w1 and w2

would be 0.5 (they have to sum to 1). However, one could make these weights dynamic.

The magnitude of the features represents the strength of boundaries, hence, a simple mean of the feature vector at location i
should be somewhat proportional to pb. Of course, fancier ways to combine the features can be explored for better performance. As a starting point, you can simply use an element-wise product of the baseline output and the mean feature strength to form the final pb value, this should work reasonably well.