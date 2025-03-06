import cv2
import sys
from matplotlib import pyplot as plt
# import pytesseract
  
  
# # Opening image
# # Testing Key
# answer_path = sys.path[0]+"/images/numbers.png"
# img = cv2.imread(answer_path, 1)
  
# # OpenCV opens images as BRG 
# # but we want it as RGB and 
# # we also need a grayscale 
# # versions
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # print(pytesseract.image_to_string(img_rgb))
# # exit()
  
# # Use minSize because for not 
# # bothering with extra-small dots 
# test_path = sys.path[0]+"/images/math_problem.png"
# find_data = cv2.CascadeClassifier(test_path)
  
# found = find_data.detectMultiScale(img_gray, 
#                                    minSize =(20, 20))
  
# # Don't do anything if there's 
# # no sign
# amount_found = len(found)
# print(amount_found)
  
# if amount_found != 0:
      
#     # There may be more than one
#     # sign in the image
#     for (x, y, width, height) in found:
          
#         # We draw a green rectangle around
#         # every recognized sign
#         cv2.rectangle(img_rgb, (x, y), 
#                       (x + height, y + width), 
#                       (0, 255, 0), 5)

# # Creates the environment 
# # of the picture and shows it
# plt.subplot(1, 1, 1)
# plt.imshow(img_rgb)
# plt.show()


# ----------------------------------
import numpy as np

# Load the image
test_path = sys.path[0]+"/images/math_problem.png"
image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

# Preprocessing
_, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(thresholded, kernel, iterations=1)

# Segmentation
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detection and Visualization
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filter contours based on size or aspect ratio if needed
    if w * h > 100:  # Example: filter small contours
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

# print(pytesseract.image_to_string(img_rgb))
# Display the result
cv2.imshow('Detected Math Problems', image)
cv2.waitKey(0)
cv2.destroyAllWindows()