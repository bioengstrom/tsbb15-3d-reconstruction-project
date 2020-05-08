import cv2
import numpy as np

# p3p using openCV
def p3p(_3d_pts, img_pts, C):
    R, t = cv2.solvePnP(_3d_pts, img_pts, C, cv2.SOLVEPNP_ITERATIVE)
    
    return R, t
