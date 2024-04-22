import os
import cv2
from Stitcher import Stitcher

# Read image resources
# imageA = cv2.imread(r"CV-image/test/2/1.jpeg")
# if imageA is None:
#     print(f"Failed to load image")
# else:
#     print(f"Successfully loaded image")
# imageB = cv2.imread(r"CV-image/test/2/2.jpeg")
# imageC = cv2.imread(r"CV-image/test/2/3.jpeg")
# # imageD = cv2.imread(r"CV-image/ImageSet5/S5-4.jpg")
#
# # images = [imageA, imageB]
# # ImageSet1
# # images = [imageC, imageB, imageA, imageD]
# # ImageSet2
# # images = [imageA, imageC, imageB]
# # ImageSet3
# images = [imageC, imageB, imageA]
# # ImageSet4
# # images = [imageA, imageC, imageB, imageD]
# # ImageSet5
# # images = [imageC, imageB, imageA, imageD]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: '{img_path}' couldn't be opened or was not an image.")
    return images

# load images
images_path = 'CV-image/test/2'
images = load_images_from_folder(images_path)

# Stitch images into panoramas
stitch = Stitcher()
print("Start Stitching ...")

# display key points matching outcome
show_matches = True
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
