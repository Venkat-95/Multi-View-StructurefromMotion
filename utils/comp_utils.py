import numpy as np
import cv2



FEATURE_EXTRACTOR = cv2.SIFT_create()
bf_matcher = cv2.BFMatcher()

def extract_features(img:np.ndarray):
    keypoints, descriptors = FEATURE_EXTRACTOR.detectAndCompute(img, None)
    return keypoints, descriptors

def match_keypoints(keypts1, desc1, keypts2, desc2):
    matches = bf_matcher.knnMatch(desc1, desc2, k=2)
    matches_passing_loewe = []
    for m, n in matches:
        if m.distance < 0.99 * n.distance:
            matches_passing_loewe.append(m)
    src_pts = np.float32([keypts1[m.queryIdx].pt for m in matches_passing_loewe]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypts2[m.trainIdx].pt for m in matches_passing_loewe]).reshape(-1, 1, 2)
    return matches_passing_loewe, src_pts, dst_pts

def compute_homography(src_pts, dst_pts, RANSAC_THRESHOLD):
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)

    return H, mask

def triangulate_points(K1, K2, R_first_frame, T_first_frame, R_next_frame, T_next_frame, src_pts, dst_pts):
    projection_mat1_temp = np.hstack((R_first_frame, T_first_frame))
    projection_mat1 = K1 @ projection_mat1_temp

    projection_mat2_temp = np.hstack((R_next_frame, T_next_frame))
    projection_mat2 = K2 @ projection_mat2_temp

    point_cloud_4D = cv2.triangulatePoints(projection_mat1, projection_mat2, src_pts, dst_pts)

    point_3d = point_cloud_4D[:3] / point_cloud_4D[3]
    point_3d = point_3d.T

    return projection_mat1, projection_mat2, point_cloud_4D, point_3d


def reprojection_error(obj_points, image_points, rotational_matrix, translational_vector, K):

    cords_3d = cv2.convertPointsFromHomogeneous(obj_points.T).reshape(-1, 3)
    dist_coeffs = np.zeros(4)

    rotational_vector, _ = cv2.Rodrigues(rotational_matrix)

    image_points_from_3d, _ = cv2.projectPoints(cords_3d, rotational_vector, translational_vector, K, dist_coeffs)
    
    image_points_from_3d = np.array(image_points_from_3d).reshape(-1, 2)
    image_points = np.array(image_points).reshape(-1, 2)

    total_error = cv2.norm(image_points_from_3d, np.float32(image_points), cv2.NORM_L2)
    avg_error = total_error / len(image_points)
    
    return avg_error, cords_3d


def computePnP(obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:

    _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)

    rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

    return rot_matrix, tran_vector, image_point, obj_point, rot_vector




def get_3D_2D_correspondences(
    filtered_matches,
    kp_curr,
    prev_2D_filtered_matches,
    prev_3D_points
):
    trainIdx_to_3d_idx = {
        m.trainIdx: idx
        for idx, m in enumerate(prev_2D_filtered_matches)
    }

    common_3d_pts = []
    corresponding_2d_pts = []

    for m in filtered_matches:
        q, t = m.queryIdx, m.trainIdx

        if q in trainIdx_to_3d_idx:
            idx3d = trainIdx_to_3d_idx[q]
            common_3d_pts.append(prev_3D_points[idx3d])
            corresponding_2d_pts.append(kp_curr[t].pt)

    if len(common_3d_pts) < 4:
        print("[PnP] Not enough valid 3D–2D correspondences:", len(common_3d_pts))
        return None, None

    pts3d = np.array(common_3d_pts, dtype=np.float32)
    pts2d = np.array(corresponding_2d_pts, dtype=np.float32)
    return pts3d, pts2d

