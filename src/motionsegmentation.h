/*
 * Robust motion segmentation
 * by Jakub Vojvoda, vojvoda@swdeveloper.sk
 * 2016
 *
 * licence: GNU LGPL v3
 * file: motionsegmentation.h
 *
 */
 
#ifndef MOTIONSEGMENTATION_H
#define MOTIONSEGMENTATION_H

#include <opencv2/highgui/highgui.hpp>

class MotionSegmentation
{
public:
    MotionSegmentation();

    void setDiffHistorySize(unsigned int s);
    void setMotionBufferSize(unsigned int s);

    cv::Mat segment(cv::Mat actual, double thresh, double wdf, double wop, double wgm, double wda);
    cv::Mat computeMask(cv::Mat segmentation, int close_winsize, int dilation_winsize, double min_area);

protected:
    cv::Mat compDifference(cv::Mat actual, double thresh);
    cv::Mat compDenseOpticalFlow(cv::Mat actual, int winsize);
    cv::Mat calcGradientMotion(cv::Mat actual, double thresh, float duration, float dmin, float dmax);
    cv::Mat compDifferenceAverage(cv::Mat actual, int winsize, double thresh, double k);

private:
    cv::Mat prev;

    unsigned int diff_history_size;
    std::vector<cv::Mat> diff_history;

    cv::Mat motion_history;
    unsigned int motion_buffer_size;
    std::vector<cv::Mat> motion_buffer;

    std::vector<double> weights;
    std::vector<cv::Mat> accumulator;

    cv::Mat compBackAverage(cv::Mat actual, cv::Mat &acc, double alpha);
    cv::Mat substractBack(cv::Mat actual, cv::Mat back);

    void resetDiffHistory();
    void resetMotionBuffer();
};

#endif // MOTIONSEGMENTATION_H
