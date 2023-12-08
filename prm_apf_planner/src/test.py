import cv2
import numpy as np
from utils import APF
from matplotlib import pyplot as plt
import time

file_path = "/home/anisha/sick/planner_ws/src/PRM-Blended-Potential-Field-Path-Planning/prm_apf_planner/map/map.png"

grid_map = cv2.imread(file_path, 0)
print(grid_map.shape)
print(grid_map)

# cv2.imshow("hi", grid_map)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

thresh = 127
im_bw = cv2.threshold(
    grid_map,
    thresh,
    255,
    cv2.THRESH_BINARY)[1]

kernel = np.ones((3, 3), np.uint8)
im_bw = cv2.erode(im_bw, kernel, iterations=1)
im_bw = cv2.copyMakeBorder(
            im_bw, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
im_bw = cv2.rotate(im_bw, cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow("hi", im_bw)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

inflated_map = im_bw
influence_coefficient = 100
repulsion_range = 5
print("here")
t1 = time.time()
apf = APF(influence_coefficient, repulsion_range)
sampling_points, samples_obs_region, samples_open_region = apf.sampling(
    inflated_map)
print("apf done")
t2 = time.time()
print("time taken by apf : ", t2-t1)
print(im_bw.shape)
print(samples_obs_region)
print(samples_open_region)
new = np.zeros(im_bw.shape)
for i in samples_obs_region:
    new[i[0], i[1]] = 40
for i in samples_open_region:
    new[i[0], i[1]] = 200
plt.imshow(new)
plt.colorbar()
plt.show()
