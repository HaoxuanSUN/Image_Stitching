import numpy as np
import cv2

class Stitcher:

    # stitch function
    def stitchs_batch_images(self, images, showMatches=False):
        """
        :param images:list[image...]
        :return:result:outcome(stitched image)
        """
        for i in range(len(images)-1):
            if i == 0:
                result1 = self.stitch_two_images([self.resize(images[i]), self.resize(images[i+1])], showMatches=showMatches)
                result2 = self.stitch_two_images([self.resize(images[i+1]), self.resize(images[i])], showMatches=showMatches)
                if result1[0].sum() > result2[0].sum():
                    result = result1
                else:
                    result = result2
            if i > 0:
                result1 = self.stitch_two_images([self.resize(result[0]), self.resize(images[i+1])], showMatches=showMatches)
                result2 = self.stitch_two_images([self.resize(images[i+1]), self.resize(result[0])], showMatches=showMatches)
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

        # Detect SIFT key feature points of A and B pictures, and calculate feature descriptors
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Match all feature points of two pictures and return the matching result
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, ransacReprojThreshold)

        # If the returned result is empty and there is no matching feature point, exit the algorithm
        if M is None:
            return None

        # Otherwise, extract matching results
        # H is a 3x3 perspective transformation matrix, A-->B
        (matches, H, status) = M
        # Transform the perspective of ImageA, ImageA_Transform is the transformed image
        ImageA_Transform = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # Convert ImageA_Transform to np.uint8 format for subsequent bit operations
        ImageA_Transform = ImageA_Transform.astype(np.uint8)
        # Use a mask to pass imageB to the leftmost of the result picture
        imageB_Transfoorm = np.zeros(ImageA_Transform.shape, dtype=np.uint8)
        imageB_Transfoorm[:imageB.shape[0], :imageB.shape[1]] = imageB

        rows, cols, channels = imageB_Transfoorm.shape
        roi = ImageA_Transform[0:rows, 0:cols]
        cv2.imshow("roi", roi)

        img2gray = cv2.cvtColor(imageB_Transfoorm, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow("mask", mask)
        mask_inv = cv2.bitwise_not(mask)

        cv2.imshow("mask_inv", mask_inv)

        img1_bg = cv2.bitwise_and(ImageA_Transform, ImageA_Transform, mask=mask_inv)
        cv2.imshow("img1_bg", img1_bg)

        img2_fg = cv2.bitwise_and(imageB_Transfoorm, imageB_Transfoorm, mask=mask)
        cv2.imshow("img2_fg", img2_fg)

        dst = cv2.add(img1_bg, img2_fg)
        ImageA_Transform[0:rows, 0:cols] = dst

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
