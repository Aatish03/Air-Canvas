# Air-Canvas

Computer vision project implemented with OpenCV

Ever wanted to draw your imagination by just waving your finger in the air? In this project, we will learn to build an Air Canvas which can draw anything on it by just capturing the motion of a coloured marker with a camera. Here a coloured object at the tip of the finger is used as the marker.

We will be using the computer vision techniques of OpenCV to build this project. The preferred language is Python due to its exhaustive libraries and easy-to-use syntax but understanding the basics it can be implemented in any OpenCV-supported language.

Here Colour Detection and tracking are used in order to achieve the objective. The color marker is detected and a mask is produced. It includes the further steps of morphological operations on the mask produced: Erosion and Dilation. Erosion reduces the impurities present in the mask and dilation further restores the eroded main mask.

# Algorithm
1. Start reading the frames and convert the captured frames to HSV color space.(Easy for color detection)
2. Prepare the canvas frame and put the respective ink buttons on it.
3. Adjust the trackbar values to find the mask of the colored marker.
4. Preprocess the mask with morphological operations.(Erotion and dilation)
5. Detect the contours, find the center coordinates of the largest contour, and keep storing them in the array for successive frames. (Arrays for drawing points on canvas)
Finally, draw the points stored in an array on the frames and canvas.
**Requirements:** python3, numpy, and opencv installed on your system.
