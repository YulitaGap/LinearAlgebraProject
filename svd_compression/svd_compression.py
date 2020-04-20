import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

# Transform image to grayscale
img = Image.open('example.jpg')
imggray = img.convert('LA')

# Show grayscaled image form
plt.figure(figsize=(9, 6))
plt.imshow(imggray)
plt.show()

# Transform image to numpy matrix
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)

#SVD decomposition
U, sigma, V = np.linalg.svd(imgmat)


#How resulting image looks using from 5-50 vectors for reconstruction after decomposition
for i in range(5, 55, 5):
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()
