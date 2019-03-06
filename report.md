
#Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort1.png "Undistorted"

[image2]: ./output_images/undistort2.png "Road Transformed"
[image3]: ./output_images/edge.png "Binary Example"
[image4]: ./output_images/warp_1.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./output_videos/output_project_video.mp4 "Video"

[result_image1]: ./output_images/result_1.png "Result1"
[curve_grad]:./output_images/curve_grad.png "curve_grad"


<!---
![alt Text](output.gif)
-->

---

### Camera Calibration

I use the OpenCV functions findChessboardCorners and drawChessboardCorners to 
get the coordinates of corners on a series of image of a chessboard taken from different angles.
Once I have all the coordinates from each image, I am able to compute the camera 
calibration matrix and distortion coefficients using the cv2.calibrateCamera() function.

At this stage, I can use `cv2.undistort()` function to correct images with same calibration matrix and distortion coefficients


Result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  


### Gradient absolute value
For absolute gradient value we simply apply a threshold to `cv2.Sobel()` output for each axis.

```python
sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
```

### Gradient magnitude
Additionaly we include pixels within a threshold of the gradient magnitude.

```python
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
```

### Gradient direction
We also include pixels that happen to be withing a threshold of the gradient direction.

```python
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
```

### Color
Finally, we extract S channel of image representation in the HLS color space and then apply a threshold on its absolute value.

```python
hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
s_channel = hls[:, :, 2]
```
Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


First step is to define source points.
```python
        (h, w) = (undist.shape[0], undist.shape[1])
        src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
        # Define corresponding destination points
        dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
```

This resulted in the following source and destination points:
src
[[ 564.  450.]
 [ 716.  450.]
 [-100.  720.]
 [1380.  720.]]
 dst
[[ 100.    0.]
 [1180.    0.]
 [ 100.  720.]
 [1180.  720.]]
 
The transformation is applied using cv2.getPerspectiveTransform() function. I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

It is implemented in birds_eye method in carmrea.py


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

the code searches the  frame from bottom to top trying to find the pixels that could be representing 
lane line boundaries. The code is trying to identify two lines  
that can be lane boundaries. For each of those lines, we have a set of 
windows. We scan the frame with those windows, 
collecting non-zero pixels within window bounds. Once we reach the top, 
we try to fit a second order polynomial into collected points. 
This polynomial coefficients would represent a single lane boundary.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `radius_of_curvature` method in my code in `Laneline.py`

From [this tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php), I can calulcate the radius of curvature in an arbitrary point using the following equation.

![alt text][curve_grad]


so, the python code is implemented as follows.
```python
# Fit a new polynomial in real world coordinate space
poly_coef = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
radius = ((1 + (2 * poly_coef[0] * 720 * ym_per_pix + poly_coef[1]) ** 2) ** 1.5) / np.absolute(2 * poly_coef[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `LaneDetect.py` in the function `draw_lane_overlay()`.  Here is an example of my result on a test image:

![alt text][result_image1]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

It will fail in many situation, for example changing weather conditions, lack or damage of lane marking.
We can extract lof of useful information by examing the gradients and color space of pixel values. These
data can be used as input of more sophisticated algorithms.

