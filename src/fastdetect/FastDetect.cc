#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char *argv[])
{
  cv::Mat img = cv::imread("../img/test.jpg", 1);
  if(img.empty()) return -1; 

  cv::Mat gray_img;
  cv::cvtColor(img, gray_img, CV_BGR2GRAY);
  cv::normalize(gray_img, gray_img, 0, 255, cv::NORM_MINMAX);

  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::KeyPoint>::iterator itk;

  // FAST 検出器 + Grid アダプタ に基づく特徴点検出
  // maxTotalKeypoints=200, gridRows=10, gridCols=10
  // TODO: 下の記法は3.4ではダメ
  cv::GridAdaptedFeatureDetector detector(new cv::FastFeatureDetector(10), 200, 10, 10);
  cv::Scalar color(100,255,100);
  detector.detect(gray_img, keypoints);
  for(itk = keypoints.begin(); itk!=keypoints.end(); ++itk) {
    cv::circle(img, itk->pt, 1, color, -1);
    cv::circle(img, itk->pt, itk->size, color, 1, CV_AA);
    if(itk->angle>=0) {
      cv::Point pt2(itk->pt.x + cos(itk->angle)*itk->size, itk->pt.y + sin(itk->angle)*itk->size);
      cv::line(img, itk->pt, pt2, color, 1, CV_AA);
    }
  }

  cv::namedWindow("GridAdapted Features", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("GridAdapted Features", img);
  cv::waitKey(0);  
}
