import numpy as np
import cv2 as cv

def convert_to_points(im) -> np.ndarray:
    # Convert to grayscale
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", imgray)
    cv.waitKey(0)

    # Threshold the image to black and white
    _, imthresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow("thresh", imthresh)
    cv.waitKey(0)

    # Thinning the image to make line segments 1 pixel wide (in general)
    imthinned = cv.ximgproc.thinning(imthresh)
    cv.imshow("thinned", imthinned)
    cv.waitKey(0)

    # Find the coordinates of the white pixels
    points = cv.findNonZero(imthinned)

    if points is not None:
        points = points.reshape(-1, 2)  # Reshape to Nx2 array of (x, y) coordinates

    return points

im = cv.imread("images/heart.jpg")
points = convert_to_points(im)
for point in points:
    cv.circle(im, tuple(point), 2, (0, 255, 0), -1)
cv.imshow("points", im)
cv.waitKey(0)
