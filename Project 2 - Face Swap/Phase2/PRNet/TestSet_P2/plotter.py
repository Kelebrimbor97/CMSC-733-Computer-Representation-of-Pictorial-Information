import matplotlib.pyplot as plt


im1 = plt.imread('Rambo.jpg')
im2 = plt.imread('Scarlett.jpg')
im3 = plt.imread('rambett.jpg')
im4 = plt.imread('Scarbo.jpg')

plt.figure()

plt.subplot(2,2,1)
plt.axis("off")
plt.imshow(im1)

plt.subplot(2,2,2)
plt.axis("off")
plt.imshow(im2)

plt.subplot(2,2,3)
plt.axis("off")
plt.imshow(im3)

plt.subplot(2,2,4)
plt.axis("off")
plt.imshow(im4)

plt.show()