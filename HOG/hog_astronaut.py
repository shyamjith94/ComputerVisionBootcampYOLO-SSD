from skimage import data, feature
import matplotlib.pyplot as plt

# fetch data of astronaut
image = data.astronaut()
print(image.shape)
hog_vector, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                    block_norm="L2", visualize=True, multichannel=True)

figure, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))

# let plot first image
axes[0].imshow(image)
axes[0].set_title("original image")

# show the HOG related image
axes[1].imshow(hog_image)
axes[1].set_title("hog image")
plt.show()