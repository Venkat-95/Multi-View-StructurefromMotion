/*
C++ version is WIP and not tested fully. 
*/

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/calib3d.hpp>
#include<vector>
#include<iostream>
#include<dirent.h>

cv::Mat cam0_K = (cv::Mat_<double>(3,3) << 3411.42, 0, 3116.72,
                                            0, 3410.02, 2062.52,
                                            0, 0, 1);

cv::Mat cam1_K = (cv::Mat_<double>(3,3) << 3409.58, 0, 3115.16,
                                            0, 3409.44, 2064.73,
                                            0, 0, 1);
cv::Mat cam2_K = (cv::Mat_<double>(3,3) << 3407.41, 0, 3112.83,
                                            0, 3408.08, 2065.6,
                                            0, 0, 1);
cv::Mat cam3_K = (cv::Mat_<double>(3,3) << 3408.35, 0, 3114.7,
                                            0, 3408.8, 2070.92,
                                            0, 0, 1);
cv::Ptr<cv::SIFT> siftptr = cv::SIFT::create();

cv::BFMatcher bfMatcher(cv::NORM_L2);



double SCALING_FACTOR = 1.0;
const float loewe_threshold = 0.65f;


std::vector<std::string> list_all_images(std::string data_file_path){
    std::vector<std::string> all_files;
    DIR* dir = opendir(data_file_path.c_str());
    if(dir == nullptr){
        std::cerr << "Error opening directory: "<<data_file_path<<std::endl;
        return all_files;
    }

    struct dirent* entry;

    while((entry = readdir(dir)) != nullptr){
        std::string file_name = entry->d_name;

        if(file_name == "." || file_name==".."){
            continue;
        }

        std::string file_path = data_file_path +"/"+file_name;
        all_files.push_back(file_path);
    }

    return all_files;
}

void computeEssentialMatrix(const std::vector<cv::Point2f>& src_pts,
                            const std::vector<cv::Point2f>& dst_pts,
                            const cv::Mat& K,
                            
                            cv::Mat& E,
                            cv::Mat& mask ){

                                E = cv::findEssentialMat(src_pts,dst_pts, K, cv::RANSAC, 0.99, 1.0,mask);
                            }


void recoverPose(const cv::Mat& E,
                 const std::vector<cv::Point2f>& src_pts,
                 const std::vector<cv::Point2f>& dst_pts,
                 const cv::Mat& K,
                 cv::Mat& R,
                 cv::Mat& T,
                 cv::Mat& mask){
                    cv::recoverPose(E,src_pts,dst_pts,K,R,T,mask);
                 }

void triangulatePoints(const cv::Mat& K,
                       const cv::Mat& R,
                       const cv::Mat& T,
                       const std::vector<cv::Point2f>& src_pts,
                       const std::vector<cv::Point2f>& dst_pts,
                       cv::Mat& projectionMatrix1,
                       cv::Mat& projectionMatrix2,
                       cv::Mat& points3D){
                        cv::Mat projectionMatrix1 = cv::Mat::zeros(3,4, CV_64F);
                        K.copyTo(projectionMatrix1(cv::Rect(0,0,3,3)));

                        cv::Mat projectionMatrix2 = cv::Mat::zeros(3,4, CV_64F);
                        R.copyTo(projectionMatrix2(cv::Rect(0,0,3,3)));
                        T.copyTo(projectionMatrix2(cv::Rect(3,0,1,3)));

                        projectionMatrix2 = K * projectionMatrix2;

                        cv::triangulatePoints(projectionMatrix1,projectionMatrix2,src_pts,dst_pts,points3D);

                        for (size_t i = 0; i < points3D.cols; i++)
                        {
                            points3D.col(i) /= points3D.at<float>(3,i);
                        }
                        
                       }
void convertTo3DPoints(const cv::Mat& points_all, std::vector<cv::Point3f>& points3D){
    points3D.clear();
    for (int i = 0; i < points_all.cols; i++)
    {
        cv::Point3f pt;
        pt.x = points_all.at<float>(0,i);
        pt.y = points_all.at<float>(1,i);
        pt.z = points_all.at<float>(2,i);

        points3D.push_back(pt);
    }
    
}

void transformation(const cv::Mat& R, const cv::Mat& T, cv::Mat& global_R, cv::Mat& global_T){
    global_T = global_T + global_R*T;
    global_R = R* global_R;
}

void reprojectionError(cv::Mat& points3D,
                        std::vector<cv::Point2f>& dst_pts,
                        cv::Mat rotationMatrix,
                        cv::Mat translationVector,
                        const cv::Mat& K ){
                            
                        }

int main(){

    std::string path_to_imgs = "../data/images/";
    std::vector<std::string> all_images_list = list_all_images(path_to_imgs);
    
    cv::Mat global_R = cv::Mat::eye(3,3, CV_64F);
    cv::Mat global_T = cv::Mat::zeros(3,1, CV_64F);

    std::vector<cv::Point3f> global_3D_points;

    for (size_t i = 0; i < all_images_list.size(); i++){
        
        cv::Mat img1_resize;
        std::vector<cv::KeyPoint> keypoint1;
        cv::Mat descriptor1;

        cv::Mat image1 = cv::imread(all_images_list[i]);
        cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
        cv::resize(image1, img1_resize, cv::Size(static_cast<int>(image1.size().width*(1/SCALING_FACTOR)), 
                                                        static_cast<int>(image1.size().height*(1/SCALING_FACTOR))), cv::INTER_CUBIC);
        
        siftptr->detectAndCompute(img1_resize,cv::noArray(),keypoint1,descriptor1);

        std::vector<std::vector<cv::DMatch>> knnMatches;
        std::cout<<"Computing Matching between "<<all_images_list[i]<< "and" <<all_images_list[j]<<std::endl;
        
        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptors2;
        cv::Mat image2 = cv::imread(all_images_list[i+1]);
        cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
        cv::Mat img2_resize;
        cv::resize(image2, img2_resize, cv::Size(static_cast<int>(image2.size().width*(1/SCALING_FACTOR)), 
                                                static_cast<int>(image2.size().height*(1/SCALING_FACTOR))), cv::INTER_CUBIC);

        
        siftptr->detectAndCompute(img2_resize, cv::noArray(), keypoints2,descriptors2);
        
        bfMatcher.knnMatch(descriptor1,descriptors2, knnMatches, 2);
        std::vector<cv::DMatch> goodMatches;
        for (const auto& knnMatch : knnMatches)
        {
            if (knnMatch.size()==2){
                const cv::DMatch& bestMatch = knnMatch[0];
                const cv::DMatch& secondbest = knnMatch[1];

                if(bestMatch.distance < loewe_threshold*secondbest.distance){
                    goodMatches.push_back(bestMatch);
                }
            }
        }
        
        std::cout<<goodMatches.size()<<std::endl;
        
        
        std::vector<cv::Point2f> src;
        std::vector<cv::Point2f> dst;
        for (size_t kp_idx = 0; kp_idx < goodMatches.size(); kp_idx++)
        {
            src.push_back( keypoint1[goodMatches[kp_idx].queryIdx].pt);
            dst.push_back(keypoints2[goodMatches[kp_idx].trainIdx].pt);
        }
        std::vector<uchar> inlierMask;
        cv::Mat H = cv::findHomography( src,dst, cv::RANSAC, 5.0, inlierMask);
        std::vector<cv::DMatch> matches_passing_homography;
        for (size_t inlier_idx = 0; inlier_idx < inlierMask.size(); inlier_idx++)
        {
            if(inlierMask[inlier_idx]){
                matches_passing_homography.push_back(goodMatches[inlier_idx]);
            }
        }
        cv::Mat output;
        cv::drawMatches(img1_resize, keypoint1, img2_resize, keypoints2, matches_passing_homography, output, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::resize(output, output, cv::Size(static_cast<int>(image2.size().width*(0.25)), 
                                                static_cast<int>(image2.size().height*(0.25))), cv::INTER_CUBIC);
        cv::imshow("Good Matches", output);
        cv::waitKey(0);
        std::vector<cv::Point2f> src_gm_corrected;
        std::vector<cv::Point2f> dst_gm_corrected;
        for (size_t gm_idx = 0; gm_idx < matches_passing_homography.size(); gm_idx++)
        {
            src_gm_corrected.push_back(keypoint1[matches_passing_homography[gm_idx].queryIdx].pt);
            dst_gm_corrected.push_back(keypoints2[matches_passing_homography[gm_idx].trainIdx].pt);
        }

        cv::Mat E, mask;
        computeEssentialMatrix(src_gm_corrected, dst_gm_corrected, cam0_K, E,mask);

        cv::Mat R,T;
        

        recoverPose(E, src_gm_corrected, dst_gm_corrected, cam0_K, R, T, mask);

        cv::Mat points4D;
        cv::Mat projectionMatrix1 = cv::Mat::zeros(3,4, CV_64F);
        cv::Mat projectionMatrix2 = cv::Mat::zeros(3,4, CV_64F);
        triangulatePoints(cam0_K, R, T, src_gm_corrected, dst_gm_corrected, projectionMatrix1, 
                            projectionMatrix2, points4D);

        std::vector<cv::Point3f> points3D;
        convertTo3DPoints(points4D,points3D);

        global_3D_points.insert(global_3D_points.end(), points3D.begin(), points3D.end());

        transformation(R,T, global_R, global_T);
        std::cout<<R<<T<<std::endl;

}

return 0;
}



