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

protected:
    cv::Mat compDifference(cv::Mat actual, double thresh);
    cv::Mat compDenseOpticalFlow(cv::Mat actual, int winsize);
    cv::Mat calcGradientMotion(cv::Mat actual, double thresh, float duration, float dmin, float dmax);
    cv::Mat compDifferenceAverage(cv::Mat actual, int winsize, double thresh, double k);

private:
    unsigned int diff_history_size;
    std::vector<cv::Mat> diff_history;

    cv::Mat prev;

    cv::Mat motion_history;
    unsigned int motion_buffer_size;
    std::vector<cv::Mat> motion_buffer;

    cv::Mat compBackAverage(cv::Mat actual, double alpha);
    cv::Mat substractBack(cv::Mat actual, cv::Mat back);

    void resetDiffHistory();
    void resetMotionBuffer();
};

#endif // MOTIONSEGMENTATION_H
