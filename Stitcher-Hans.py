import numpy as np
import cv2
from sympy import centroid
from tqdm import tqdm
import json

class Stitcher:

    # stitch function
    def stitchs_batch_images(self, images, showMatches=False):
        """
        :param images:list[image...]
        :return:result:outcome(stitched image)
        """
        # for i in range(len(images)):
        # # Adjust brightness and contrast of image2 to match image1
        #     images[i] = adjust_brightness_contrast(images[i], brightness=30, contrast=20)

        for i in range(len(images)-1):
            if i == 0:
                result = self.stitch_two_images([ normal_resize(images[i]),  normal_resize(images[i+1])], showMatches=showMatches)
                # result2 = self.stitch_two_images([images[i+1], images[i]], showMatches=showMatches)
                # if result1[0].sum() > result2[0].sum():
                #     result = result1
                # else:
                #     result = result2
            if i > 0:
                result1 = self.stitch_two_images([result[0],  normal_resize(images[i+1])], showMatches=showMatches)
                result2 = self.stitch_two_images([ normal_resize(images[i+1]), result[0]], showMatches=showMatches)
                if result1[0].sum() > result2[0].sum():
                    result = result1
                else:
                    result = result2

        
        return result

    def resize(self, image, image_h=500, ratio=3):
        """
        :param image:
        :param image_h: Zoom the picture to the same height, image_h cannot be too large, otherwise the calculation time will be too long
        :param ratio: ratio>1 Expand the edge of the picture to prevent the picture from being missed during spatial mapping. The larger the ratio, the less likely being missed
        :return:
        """
        # get height/width/chanel info of image
        height, width, chanel = image.shape
        # resized_height = image_h
        resize_h = image_h
        # resized_width = image width * image_h/image height
        resize_w = int(width * resize_h/height)
        # resize images: interpolation algorithm using bicubic interpolation
        image_resize = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
        image_expand = np.zeros((int(ratio*resize_h), int(ratio*resize_w), chanel), dtype=np.uint8)
        image_expand[resize_h//2:resize_h//2+resize_h, resize_w//2:resize_w//2+resize_w] = image_resize
        return image_expand

    def get_max_rectangle(self, img):
        # Get the largest circumscribed rectangle of the picture content, filter out the filled part
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imggray, 10, 255, 0)

        #_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # boundary rectangle
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w]

    def stitch_two_images(self, images, ratio=0.75, ransacReprojThreshold=4.0, showMatches=False):
        """
        Feature point matching and picture stitching for 2 images
        :param images:
        :param ratio:
        :param ransacReprojThreshold:
        :param showMatches:
        :return:
        """
        # Obtain the input pictures, imageB is on the left, imageA is on the right by default
        (imageB, imageA) = images

        imageB_Transfoorm = np.zeros( (imageA.shape[0] + imageB.shape[0], imageA.shape[1]+ imageB.shape[1], 3), dtype=np.uint8)
        start_x = imageB_Transfoorm.shape[0]//4
        start_y = imageB_Transfoorm.shape[1]//4
        imageB_Transfoorm[start_x: start_x+imageB.shape[0], start_y: start_y+imageB.shape[1]] = imageB
        # Detect SIFT key feature points of A and B pictures, and calculate feature descriptors
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB_Transfoorm)
        
        # Match all feature points of two pictures and return the matching result
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, ransacReprojThreshold)
        print("match finished")
        # If the returned result is empty and there is no matching feature point, exit the algorithm
        if M is None:
            return None
        

        # Otherwise, extract matching results
        # H is a 3x3 perspective transformation matrix, A-->B
        (matches, H, status) = M
        cv2.imwrite("imageA.jpg", imageA)
        # Transform the perspective of ImageA, ImageA_Transform is the transformed image
        ImageA_Transform = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]+ imageB.shape[0]), borderMode = cv2.BORDER_TRANSPARENT)
        
        # Convert ImageA_Transform to np.uint8 format for subsequent bit operations
        ImageA_Transform = ImageA_Transform.astype(np.uint8)
        # Use a mask to pass imageB to the leftmost of the result picture
        # imageB_Transfoorm = np.zeros(ImageA_Transform.shape, dtype=np.uint8)
        # start_x = imageB_Transfoorm.shape[0]//2
        # start_y = imageB_Transfoorm.shape[1]//2
        # imageB_Transfoorm[start_x: start_x+imageB.shape[0], start_y: start_y+imageB.shape[1]] = imageB
        
        rows, cols, channels = imageB_Transfoorm.shape
        roi = ImageA_Transform[0:rows, 0:cols]

        img2gray = cv2.cvtColor(imageB_Transfoorm, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(ImageA_Transform, ImageA_Transform, mask=mask_inv)
        img2_fg = cv2.bitwise_and(imageB_Transfoorm, imageB_Transfoorm, mask=mask)
        cv2.imwrite("warp1.jpg", ImageA_Transform)
        cv2.imwrite("warp2.jpg", imageB_Transfoorm)

        centroid = calculate_centroid(cv2.add(img1_bg, img2_fg))
        print(centroid)
        
        # Stitching procedure, store results in warped_l.
        warped_l = ImageA_Transform
        warped_r = imageB_Transfoorm
        centroid = calculate_centroid(cv2.add(img1_bg, img2_fg))
        gradient_center = (imageB.shape[0] + (imageB.shape[0] - centroid[0]), imageB.shape[1] + (imageB.shape[1] - centroid[1]))
        print(gradient_center)
        max_distance =  np.sqrt((gradient_center[0] - centroid[0]) ** 2 + (gradient_center[1] - gradient_center[1]) ** 2)*2

        for i in tqdm(range(warped_r.shape[0])):
            for j in range(warped_r.shape[1]):
                pixel_l = warped_l[i, j, :]
                pixel_r = warped_r[i, j, :]

                
                if is_close_to_black(pixel_l):
                    if is_close_to_black(pixel_r):
                        warped_l[i, j, :] = 0.5*pixel_l + 0.5*pixel_r
                    else:
                        warped_l[i, j, :] = pixel_r
                else:
                    if is_close_to_black(pixel_r) :
                        warped_l[i, j, :] = pixel_l
                    else:
                        warped_l[i, j, :]  = gradient_blend(pixel_l, pixel_r, gradient_center, i, j, max_distance)
                        #warped_l[i, j, :] = 0.5*pixel_l + 0.5*pixel_r

        dst = warped_l
        #dst = cv2.add(img1_bg, img2_fg)
        cv2.imshow("dst", warped_l)
        ImageA_Transform[0:rows, 0:cols] = dst

        # Blending the warped image with the second image using alpha blending
       

        result = self.get_max_rectangle(ImageA_Transform)
        cv2.imshow('res', result)

        # Check whether it is necessary to display key points matching outcome
        vis = None
        if showMatches:
            # Generate points matching picture
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        # return matching results
        return [result, vis]

    def cv_show(self, name, img):
        cv2.imshow(name, img)

    def detectAndDescribe(self, image):
        # set up a SIFT generator
        # descriptor = cv2.xfeatures2d.SIFT_create()
        descriptor = cv2.SIFT_create()
        # Detect SIFT feature points and calculate descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)
        # Convert the coordinates of the feature points into a NumPy array form, otherwise you get a list instead of an array
        kps = np.float32([kp.pt for kp in kps])
        # Return feature point set and corresponding description features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, ransacReprojThreshold):
        # Create a brute-force matcher
        matcher = cv2.BFMatcher()
        # Use KNN to detect SIFT feature matching pairs from A and B images, K=2
        raw_Matches = matcher.knnMatch(featuresA, featuresB, 2)
        # print(raw_Matches)
        matches = []
        for m in raw_Matches:
            # If the ratio of the closest distance to the next closest distance is less than the certain value, keep this matching pair
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # Store the index values of two points in featuresA and featuresB
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # When the matched pair after filtering is greater than 4, the perspective transformation matrix is calculated
        if len(matches) > 4:
            # Get coordinates of matching pairs
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # Calculate the perspective transformation matrix
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            # return outcome
            return (matches, H, status)
        # Return None if the matching pair is less than or equal to 4
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # Initialize the visualization picture and connect the A and B pictures together
        (heightA, weightA) = imageA.shape[:2]
        (heightB, weightB) = imageB.shape[:2]
        vis = np.zeros((max(heightA, heightB), weightA + weightB, 3), dtype="uint8")
        vis[0:heightA, 0:weightA] = imageA
        vis[0:heightB, weightA:] = imageB
        # Traversal to draw all matching pairs
        for ((trainIdx, queryIdx), n) in zip(matches, status):
            # When the point pair matching is successful, draw on the visualization
            if n == 1:
                # Draw matching pairs
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + weightA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # Return visualization results
        return vis


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    # Brightness adjustment
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        # Apply brightness adjustment
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    # Contrast adjustment
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        # Apply contrast adjustment
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image

def is_close_to_black(pixel, threshold= 30):
    black = np.array([0,0,0])  # Black pixel
    distance = np.linalg.norm(pixel - black)  # Euclidean distance
    return distance < threshold

def normal_resize(image, scale_percent = 20):

    # Calculate the new width and height based on the scaling factor
    new_width = int(image.shape[1] * scale_percent / 100)
    new_height = int(image.shape[0] * scale_percent / 100)

    # Resize the image using the calculated width and height
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def gradient_blend(pixel_l, pixel_r, centroid, x, y, max_distance):
    max_alpha = 0.9
    center_x = centroid[0]
    center_y = centroid[1]
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    alpha = 1 - (distance / max_distance) * 1.1
    # if(distance < 10):
    #     return np.array([255,0,0])
    if(alpha < 0):
        alpha = 0
    return pixel_r*alpha + pixel_l*(1 - alpha)


def calculate_centroid(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to obtain a binary mask
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store centroid coordinates
    centroid_x = 0
    centroid_y = 0

    # Iterate through contours
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        # Calculate centroid coordinates
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

    return centroid_y, centroid_x