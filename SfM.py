import cv2
import numpy as np
import os
import open3d as o3d
import glob

from utils.comp_utils import extract_features, compute_homography, get_3D_2D_correspondences, match_keypoints, triangulate_points, reprojection_error, computePnP 
from utils.vis_utils import get_rgb_for_keypoints, remove_outliers, draw_matches
import re

DATA = "data/door/images"

SCALE_FACTOR = 4
VISUALIZE = True

initial_frame = False
comp_pnp_flag = True

def camera_id_img_name_map(file_path):
    camera_mappings = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if line.startswith('#'):
            continue
            
        line_segments = re.split(r'\s+', line.strip())
        if len(line_segments)<10:
            continue
        
        camera_id = int(line_segments[8])
        image_name = os.path.basename(line_segments[9])

        if image_name.__contains__(".JPG"):
            if camera_id not in camera_mappings: 
                camera_mappings[camera_id] = []
            camera_mappings[camera_id].append(image_name)

    return camera_mappings

def load_intrinsic_parameter(camera_img_map, calib_text_fp):
    with open(calib_text_fp, 'r') as file:
        lines = file.readlines()

    camera_intrinsics_by_id = {}
    for line in lines:
        if line.startswith("#"):
            continue
        line_segments = re.split(r'\s+', line.strip())
        if len(line_segments) < 5:
            continue

        camera_id = int(line_segments[0])
        camera_type = line_segments[1]
        image_width = float(line_segments[2]) / SCALE_FACTOR
        image_height = float(line_segments[3]) / SCALE_FACTOR
        f_x = float(line_segments[4]) / SCALE_FACTOR
        f_y = float(line_segments[5]) / SCALE_FACTOR
        c_x = float(line_segments[6]) / SCALE_FACTOR
        c_y = float(line_segments[7]) / SCALE_FACTOR

        k_mat = np.array([
            [f_x, 0, c_x],
            [0,   f_y, c_y],
            [0,    0,  1]
        ])

        if len(line_segments) > 8:
            # Distortion coefficients should NOT be scaled
            k1 = float(line_segments[8])
            k2 = float(line_segments[9])
            t1 = float(line_segments[10])
            t2 = float(line_segments[11])
            k3 = float(line_segments[12])
            k4 = float(line_segments[13])
            sx1 = float(line_segments[14])
            sx2 = float(line_segments[15])
            dist_coeffs = np.array([k1, k2, t1, t2, k3, k4, sx1, sx2])
            camera_intrinsics_by_id[camera_id] = [camera_type, image_width, image_height, k_mat, dist_coeffs]
        else:
            camera_intrinsics_by_id[camera_id] = [camera_type, image_width, image_height, k_mat]

    return camera_intrinsics_by_id

if __name__ == "__main__":

    all_imgs = []
    matched_pairs = {}
    rgb_values = []

    img_data = {}

    dst_pt_data = {}
    point_cloud_dict = {}
    rotation_matrix_dict = {}
    translation_matrix_dict = {}

    all_imgs += sorted([img for img in glob.glob(DATA+"/*.JPG")])

    image_txt_fp = "data/door/calibration/images.txt"
    camera_txt_fp = "data/door/calibration/cameras.txt"
    camera_img_map = camera_id_img_name_map(image_txt_fp)

    camera_intrinsics_with_id = load_intrinsic_parameter(camera_img_map, camera_txt_fp)

    for idx in range(0, len(all_imgs) - 1):

        image_name_1 = os.path.basename(all_imgs[idx])
        image_name_2 = os.path.basename(all_imgs[idx + 1])
        
        
        matched_pairs[image_name_1] = image_name_2
        print(f"Computing matching between : {image_name_1} and {image_name_2}")

        img1 = cv2.imread(all_imgs[idx])
        
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1_resize = cv2.resize(img1_gray, (int(img1_gray.shape[1]/SCALE_FACTOR), int(img1_gray.shape[0]/SCALE_FACTOR)), cv2.INTER_CUBIC)

        img2 = cv2.imread(all_imgs[idx + 1])
        print(img2.shape)
        img2_col = cv2.resize(img2, (int(img2.shape[1]/SCALE_FACTOR), int(img2.shape[0]/SCALE_FACTOR)), cv2.INTER_CUBIC)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_resize = cv2.resize(img2_gray, (int(img2_gray.shape[1]/SCALE_FACTOR), int(img2_gray.shape[0]/SCALE_FACTOR)), cv2.INTER_CUBIC)
        img_data[image_name_2] = img2_resize

        kp1, desc1 = extract_features(img1_resize)
        kp2, desc2 = extract_features(img2_resize)
        
        matches_before_homography, src_pts, dst_pts = match_keypoints(
            keypts1=kp1, keypts2=kp2, desc1=desc1, desc2=desc2)

        homography_matrix, mask_homography = compute_homography(src_pts, dst_pts, 25.0)
        matchesMask = mask_homography.ravel().tolist()

        filtered_matches = [m for i, m in enumerate(matches_before_homography) if mask_homography[i] == 1]
        if len(filtered_matches)<150:
            top_match = -1
            for _image_name_2,kp_data in dst_pt_data.items():
                matches_before_homography, src_pts, dst_pts = match_keypoints(
                keypts1=kp_data[0], keypts2=kp2, desc1=kp_data[1], desc2=desc2)

                homography_matrix, mask_homography = compute_homography(src_pts, dst_pts, 50.0)
                matchesMask = mask_homography.ravel().tolist()

                _filtered_matches = [m for i, m in enumerate(matches_before_homography) if mask_homography[i] == 1]
                print(f"Observing {_image_name_2} and {image_name_2} : ", len(_filtered_matches))
                if len(_filtered_matches)>top_match and len(_filtered_matches)>100:
                    top_match = len(_filtered_matches)
                    filtered_matches = _filtered_matches
                    print(f"Found the Match with {len(filtered_matches)} Matches :",_image_name_2)
                    kp1 = kp_data[0]
                    desc1 = kp_data[1]
                    prev_2D_filtered_matches = kp_data[3]
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
                    prev_3D_points = point_cloud_dict[_image_name_2]
                    

                    R_prev = rotation_matrix_dict[_image_name_2]
                    t_prev = translation_matrix_dict[_image_name_2]

                    img1_resize = img_data[_image_name_2]
                    image_name_1 = _image_name_2

                    if VISUALIZE:
                        img3 = cv2.drawMatches(img1_resize, kp1, img2_resize, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        cv2.imshow("Matches", img3)
                        cv2.waitKey(2)
                else:
                    continue   


                    
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

        camera_id_1 = list(filter(lambda key: image_name_1 in camera_img_map[key], camera_img_map))[0]
        camera_id_2 = list(filter(lambda key: image_name_2 in camera_img_map[key], camera_img_map))[0]
  
        K1 = camera_intrinsics_with_id[camera_id_1][3]
        K2 = camera_intrinsics_with_id[camera_id_2][3]

        if len(camera_intrinsics_with_id[camera_id_2])>4:
            distortion_coeffs = camera_intrinsics_with_id[camera_id_2][4]
            src_pts = cv2.undistortImagePoints(src_pts, K1, distortion_coeffs, None)
            dst_pts = cv2.undistortImagePoints(dst_pts,K2, distortion_coeffs, None)

        else:
            distortion_coeffs = None
        
        E, mask = cv2.findEssentialMat(src_pts,dst_pts,K2, prob=0.99,threshold=0.90,mask=None)
        mask = mask.ravel()
        filtered_matches = [m for m,keep in zip(filtered_matches, mask) if keep]
        src_pts = src_pts[mask==1]
        dst_pts = dst_pts[mask==1]



        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(E, src_pts, dst_pts, K2)
        em_mask = em_mask.ravel()
        
        filtered_matches = [m for m,keep in zip(filtered_matches, em_mask) if keep]
        src_pts = src_pts[em_mask > 0]
        dst_pts = dst_pts[em_mask > 0]


        if idx==0:
            R_init = np.eye(3)
            t_init = np.zeros((3, 1))
            
            rot_matrix_frame_1 = np.matmul(R_init,rot_matrix)
            trans_matrix_frame_1 = t_init + np.matmul(R_init,tran_matrix)

            transformation_matrix_1, transformation_matrix_2 , point_cloud_4D, points_3d = triangulate_points(K1, K2, R_init, t_init,
                                                                                                               rot_matrix_frame_1, trans_matrix_frame_1,
                                                                                                                 src_pts, dst_pts)
            error, _  = reprojection_error(point_cloud_4D, dst_pts, rot_matrix_frame_1, trans_matrix_frame_1, K2)
            print("REPROJECTION ERROR: ", error)
            rgb_values_1 = get_rgb_for_keypoints(img2_col, dst_pts, SCALE_FACTOR)
            rgb_values.extend(rgb_values_1)

            R_prev = rot_matrix_frame_1
            t_prev = trans_matrix_frame_1
            points_3d_total = points_3d

            if comp_pnp_flag:
                prev_3D_points = points_3d
                point_cloud_dict[image_name_2] = points_3d
                rotation_matrix_dict[image_name_2] = rot_matrix_frame_1
                translation_matrix_dict[image_name_2] = trans_matrix_frame_1
                prev_2D_filtered_matches = filtered_matches
                dst_pt_data[image_name_2] = [kp2,desc2,dst_pts,prev_2D_filtered_matches]
                
        else:
            if not comp_pnp_flag:
                rot_matrix_next_frame = np.matmul(R_prev, rot_matrix)
                trans_matrix_next_frame = t_prev+ np.matmul(R_prev,tran_matrix)
            else: 

                filtered_3d_pts, dst_filtered_pts = get_3D_2D_correspondences(
                                                        filtered_matches,
                                                        kp2,
                                                        prev_2D_filtered_matches,
                                                        prev_3D_points
                                                    )
                                                    
                if filtered_3d_pts is not None:
                    rot_matrix_next_frame, trans_matrix_next_frame, _, _, _ = computePnP(filtered_3d_pts, dst_filtered_pts, K2, distortion_coeffs, src_pts, initial=1)

                else:
                    rot_matrix_next_frame = np.matmul(R_prev, rot_matrix)
                    trans_matrix_next_frame = t_prev + np.matmul(R_prev, tran_matrix)

            transformation_matrix_2, transformation_matrix_curr , point_cloud_4D, points_3d = triangulate_points(K1,K2, R_prev,t_prev, rot_matrix_next_frame, trans_matrix_next_frame, src_pts,dst_pts)
            error, _ = reprojection_error(point_cloud_4D, dst_pts, rot_matrix_next_frame, trans_matrix_next_frame, K2)
            print("REPROJECTION ERROR 1st comp: ", error)


            rgb_values_1 = get_rgb_for_keypoints(img2_col, dst_pts,SCALE_FACTOR)
            rgb_values.extend(rgb_values_1)

            
            proj_pts, _ = cv2.projectPoints(points_3d, rot_matrix_next_frame, trans_matrix_next_frame, K2, None)
            if VISUALIZE:
                cv2.imshow("Projected vs Observed", draw_matches(img2_col, proj_pts, dst_pts))
                cv2.waitKey(2)
            

            points_3d_total = np.append(points_3d_total,points_3d,axis=0)
            rotation_matrix_dict[image_name_2] = rot_matrix_next_frame
            translation_matrix_dict[image_name_2] = trans_matrix_next_frame

            R_prev = rot_matrix_next_frame
            t_prev = trans_matrix_next_frame
            
            
            point_cloud_dict[image_name_2] = points_3d
            dst_pt_data[image_name_2] = [kp2,desc2,dst_pts, filtered_matches]

            prev_3D_points = points_3d
            prev_2D_filtered_matches = filtered_matches
            
            

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d_total)
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb_values) / 255.0)
    point_cloud = remove_outliers(point_cloud)
    o3d.visualization.draw_geometries([point_cloud])

        