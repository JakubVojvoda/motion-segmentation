/*
 * Robust motion segmentation
 * by Jakub Vojvoda, vojvoda@swdeveloper.sk
 * 2016
 *
 * licence: GNU LGPL v3
 * file: main.cpp
 *
 */

#include "src/motionsegmentation.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " filepath" << std::endl;
        return 1;
    }

    cv::VideoCapture cap(argv[1]);

    MotionSegmentation motion;
    cv::Mat frame;

    while (1) {

        cap >> frame;

        if (frame.empty()) {
            return 1;
        }

        // TODO: command-line arguments parsing
        cv::Mat segm = motion.segment(frame, 12.0, 0.25, 0.25, 0.25, 0.25);
        cv::Mat result = motion.computeMask(segm, 7, 5, 80.0);

        cv::imshow("Motion segmentation", result);

        if (cv::waitKey(1) > 0) {
            break;
        }
    }
}



