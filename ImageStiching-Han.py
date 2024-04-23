from Stitcher import Stitcher
import cv2
# Read image resources
imageA = cv2.imread(r"CV-image/test/6/img1.jpg")
if imageA is None:
    print(f"Failed to load image")
else:
    print(f"Successfully loaded image")
imageB = cv2.imread(r"CV-image/test/6/img2.jpg")
#imageC = cv2.imread(r"CV-image/test/6/6.jpg")
# imageD = cv2.imread(r"CV-image/ImageSet5/S5-4.jpg")

# ImageSet1
# images = [imageC, imageB, imageA, imageD]
# ImageSet2
# images = [imageA, imageC, imageB]
# ImageSet3
images = [imageA, imageB]
# ImageSet4
# images = [imageA, imageC, imageB, imageD]
# ImageSet5
# images = [imageC, imageB, imageA, imageD]

# Stitch images into panoramas
stitch = Stitcher()
print("Start Stitching ...")

# display key points matching outcome
show_matches = False
result = stitch.stitchs_batch_images(images, showMatches=show_matches)

# Display all pictures
# cv2.namedWindow("Result2_ACB",0)
# cv2.resizeWindow("Result2_ACB", 640, 480)
# display the image in a window
cv2.imshow("test1_ABC", result[0])
# save image
cv2.imwrite("test1_ABC.jpg", result[0])


if show_matches:
    cv2.imshow("test1_matches", result[1])
    cv2.imwrite("test1_matches.jpg", result[1])

print("Stitching Done. Press any key to close the images.")
# Display the windows infinitely until any key is pressed
cv2.waitKey(0)
# cv2.destroyAllWindows()
