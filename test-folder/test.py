import cv2 as cv

# Load image
img = cv.imread('simcam_depth.png')

# Apply color map
img = img * 5
img = cv.applyColorMap(img, cv.COLORMAP_TWILIGHT)


# Show image
cv.imshow('Test', img)

cv.imwrite('simcam_depth_colored.png', img)
# cv.waitKey(0)