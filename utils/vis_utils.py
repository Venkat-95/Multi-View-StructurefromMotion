import numpy as np
import cv2

def get_rgb_for_keypoints(image, keypoints, scale_factor):

    rgb_values = []
    
    for kp in keypoints:
        x, y = int(kp[0][0]), int(kp[0][1])
        bgr_value = image[y, x]
        
        rgb_value = [bgr_value[2], bgr_value[1], bgr_value[0]]
        rgb_values.append(rgb_value)
    
    return rgb_values



def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_clean


def draw_matches(image, projected_points, observed_points, color_projected=(0, 255, 0), color_observed=(0, 0, 255)):

    img = image.copy()

    projected_points = np.squeeze(projected_points)
    observed_points = np.squeeze(observed_points)

    for proj_pt, obs_pt in zip(projected_points, observed_points):
        proj_x, proj_y = int(proj_pt[0]), int(proj_pt[1])
        obs_x, obs_y = int(obs_pt[0]), int(obs_pt[1])

        cv2.circle(img, (proj_x, proj_y), 3, color_projected, -1)
        cv2.circle(img, (obs_x, obs_y), 3, color_observed, -1)
        cv2.line(img, (proj_x, proj_y), (obs_x, obs_y), (255, 0, 0), 1)

    return img
