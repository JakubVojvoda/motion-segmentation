/*
 * Robust motion segmentation
 * by Jakub Vojvoda, vojvoda@swdeveloper.sk
 * 2016
 *
 * licence: GNU LGPL v3
 * file: motionsegmentation.cpp
 *
 */
 
#include "motionsegmentation.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/optflow/motempl.hpp>
#include <opencv2/video/tracking.hpp>
#include <time.h>
#include <cmath>


#ifndef M_PI
    #define M_PI 3.14159
#endif


MotionSegmentation::MotionSegmentation()
{
    double w[] = {
        0.05, 0.100, 0.150,  0.200, 0.250, 0.300, 0.350, 0.400, 0.500,
        0.50, 0.500, 0.525,  0.550, 0.575, 0.600, 0.625, 0.650, 0.675,
        0.70, 0.725, 0.750,  0.775, 0.800, 0.825, 0.850, 0.875, 0.900
    };

    weights = std::vector<double>(w, w + sizeof(w) / sizeof(double));

    for (unsigned int i = 0; i < weights.size(); i++) {
        accumulator.push_back(cv::Mat());
    }

    setDiffHistorySize(2);
    setMotionBufferSize(3);
}

void MotionSegmentation::setDiffHistorySize(unsigned int s)
{
    diff_history_size = s;
    resetDiffHistory();
}

void MotionSegmentation::setMotionBufferSize(unsigned int s)
{
    motion_buffer_size = s;
    resetMotionBuffer();
}

cv::Mat MotionSegmentation::segment(cv::Mat actual, double thresh, double wdf, double wop, double wgm, double wda)
{
    cv::Mat img = actual.clone();
    cv::blur(img, img, cv::Size(3, 3));

    cv::Mat mdiff, mopt, mgrad, mavg;

    if (wdf > 0) {
        mdiff = compDifference(img, 10.0);
        cv::normalize(mdiff, mdiff, 0, 255);
    }

    if (wop > 0) {
        cv::Mat flow = compDenseOpticalFlow(img, 12);
        cv::cvtColor(flow, mopt, CV_BGR2GRAY);
        cv::normalize(mopt, mopt, 0, 255, CV_MINMAX);
    }

    if (wgm > 0) {
        mgrad = calcGradientMotion(img, 10.0, 0.75, 0.05, 0.5);
        cv::normalize(mgrad, mgrad, 0, 255, CV_MINMAX);
    }

    if (wda > 0) {
        mavg = compDifferenceAverage(img, 3, 40.0, 1.5);
        cv::normalize(mavg, mavg, 0, 255, CV_MINMAX);
    }

    cv::Mat map = cv::Mat::zeros(actual.size(), CV_8U);

    for (int x = 0; x < map.rows; x++) {
        for (int y = 0; y < map.cols; y++) {

            double diff = (wdf > 0) ? wdf * (mdiff.at<uchar>(x,y) / 255.0) : 0.0;
            double flow = (wop > 0) ? wop * (mopt.at<uchar>(x,y)  / 255.0) : 0.0;
            double grad = (wgm > 0) ? wgm * (mgrad.at<uchar>(x,y) / 255.0) : 0.0;
            double davg = (wda > 0) ? wda * (mavg.at<uchar>(x,y)  / 255.0) : 0.0;

            map.at<uchar>(x,y) = std::pow(diff + flow + grad + davg, 2) * 255;
        }
    }

    cv::Mat mask;
    cv::normalize(map, map, 0, 255, CV_MINMAX);
    cv::threshold(map, mask, thresh, 255, CV_THRESH_BINARY);

    return mask;
}

cv::Mat MotionSegmentation::computeMask(cv::Mat segmentation, int close_winsize, int dilation_winsize, double min_area)
{
    if (segmentation.empty()) {
        return cv::Mat();
    }

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(close_winsize, close_winsize));
    cv::morphologyEx(segmentation, segmentation, cv::MORPH_CLOSE, element);

    element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilation_winsize, dilation_winsize));
    cv::dilate(segmentation, segmentation, element);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(segmentation, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    cv::Mat result = cv::Mat::zeros(segmentation.size(), CV_8U);

    for (unsigned int i = 0; i< contours.size(); i++ ) {

        if (cv::contourArea(contours.at(i)) > min_area) {
            cv::drawContours(result, contours, i, cv::Scalar(255,255,255), -1, 8, hierarchy, 0);
        }
    }

    return result.clone();
}

cv::Mat MotionSegmentation::compDifference(cv::Mat actual, double thresh)
{
    if (diff_history_size < 2 || actual.empty()) {
        return cv::Mat();
    }

    cv::Mat diff = cv::abs(actual - diff_history.at(0));
    cv::Mat binary = cv::Mat::zeros(diff.size(), CV_8U);

    for (unsigned int i = 1; i < diff_history_size - 1; i++) {
        if (!diff_history.at(i-1).empty()) {
            diff += cv::abs(diff_history.at(i-1) - diff_history.at(i));
        }
    }

    if (diff.channels() == 3) {

        for (int i = 0; i < diff.rows; i++) {
            for (int j = 0; j < diff.cols; j++) {
                if (diff.at<cv::Vec3b>(i,j)[0] < thresh ||
                    diff.at<cv::Vec3b>(i,j)[1] < thresh ||
                    diff.at<cv::Vec3b>(i,j)[2] < thresh) {

                    binary.at<uchar>(i,j) = 0;
                }
                else {
                    binary.at<uchar>(i,j) = 255;
                }
            }
        }
    }
    else if (diff.channels() == 1) {
        cv::threshold(diff, binary, thresh, 255, CV_THRESH_BINARY);
    }

    for (int i = diff_history_size - 1; i > 0; i--) {
        diff_history.at(i) = diff_history.at(i-1).clone();
    }

    diff_history.at(0) = actual.clone();
    return binary;
}

cv::Mat MotionSegmentation::compDenseOpticalFlow(cv::Mat actual, int winsize)
{
    if (actual.empty()) {
        return cv::Mat();
    }

    if (prev.empty()) {
        prev = actual.clone();
    }

    cv::Mat next_gray, prev_gray;
    cv::cvtColor(actual, next_gray, CV_BGR2GRAY);
    cv::cvtColor(prev,   prev_gray, CV_BGR2GRAY);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow,
                                 0.5, 1, winsize, 5, 5, 2.2,
                                 cv::OPTFLOW_FARNEBACK_GAUSSIAN);

    std::vector<cv::Mat> flow_channels;
    cv::split(flow, flow_channels);

    cv::Mat magn, angle;
    cv::cartToPolar(flow_channels.at(0), flow_channels.at(1), magn, angle);

    cv::Mat norm;
    cv::normalize(magn, norm, 0, 255, cv::NORM_MINMAX);

    cv::Mat hsv, bgr;
    hsv = cv::Mat::zeros(angle.size(), CV_8UC3);

    for (int i = 0; i < angle.rows; i++) {
        for (int j = 0; j < angle.cols; j++) {
            hsv.at<cv::Vec3b>(i,j)[0] = (180.0 * angle.at<float>(i,j)) / (M_PI / 2.0);
            hsv.at<cv::Vec3b>(i,j)[2] = norm.at<float>(i,j);
        }
    }

    cv::cvtColor(hsv, bgr, CV_HSV2BGR);
    prev = actual.clone();
    return bgr;
}

cv::Mat MotionSegmentation::calcGradientMotion(cv::Mat actual, double thresh, float duration, float dmin, float dmax)
{
    double stime = double(clock()) / CLOCKS_PER_SEC;

    cv::Mat actual_gray, prev_gray;
    prev_gray = motion_buffer.at(motion_buffer_size - 1);
    cv::cvtColor(actual, actual_gray, CV_BGR2GRAY);

    cv::Mat silhouette = cv::Mat::ones(actual.size(), CV_8U) * 255;

    if (!prev_gray.empty()) {
        cv::absdiff(actual_gray, prev_gray, silhouette);
    }

    cv::threshold(silhouette, silhouette, thresh, 1, CV_THRESH_BINARY);

    if (motion_history.empty()) {
        motion_history = cv::Mat::ones(silhouette.size(), CV_32F) * 255;
    }

    cv::motempl::updateMotionHistory(silhouette, motion_history, stime, duration);

    cv::Mat motion;
    double alpha = 255.0 / duration;
    double beta  = (duration - stime) * alpha;
    motion_history.convertTo(motion, CV_8U, alpha, beta);

    cv::Mat out = motion.clone();

    cv::Mat orient;
    cv::motempl::calcMotionGradient(motion_history, motion, orient, dmax, dmin, 7);

    for (int i = motion_buffer_size - 1; i > 0; i--)
        motion_buffer.at(i) = motion_buffer.at(i-1).clone();

    motion_buffer.at(0) = actual_gray.clone();

    return out.clone();
}

cv::Mat MotionSegmentation::compDifferenceAverage(cv::Mat actual, int winsize, double thresh, double k)
{
    cv::Mat result = cv::Mat::zeros(actual.size(), CV_64F);

    for (unsigned int i = 0; i < weights.size(); i++) {

        double weight = weights.at(i);

        cv::Mat blured;
        cv::blur(actual, blured, cv::Size(winsize, winsize));

        cv::Mat avg  = compBackAverage(blured, accumulator.at(i), weight);
        cv::Mat fore = substractBack(actual, avg);

        cv::Mat tmp;
        cv::pow(fore, 2, tmp);

        cv::Mat gray, bgr = tmp - actual;
        cv::cvtColor(bgr, gray, CV_BGR2GRAY);

        cv::Mat partial;
        cv::threshold(gray, partial, thresh, 255, CV_THRESH_TOZERO);

        double norm = 1.0 / 255.0;
        double w = (weight < 0.5) ? k * weight : weight;

        for (int x = 0; x < partial.rows; x++) {
            for (int y = 0; y < partial.cols; y++) {

                double value = double(partial.at<uchar>(x,y)) / double(weights.size());
                result.at<double>(x,y) += norm * w * value;
            }
        }
    }

    cv::Mat gret = cv::Mat(result.size(), CV_8U);

    for (int x = 0; x < gret.rows; x++) {
        for (int y = 0; y < gret.cols; y++) {

            double rv = result.at<double>(x,y) * result.at<double>(x,y);
            gret.at<uchar>(x,y) = 255 * rv * double(weights.size());
        }
    }


    cv::Mat normalized;
    cv::equalizeHist(gret, normalized);

    return normalized.clone();
}

cv::Mat MotionSegmentation::compBackAverage(cv::Mat actual, cv::Mat &acc, double alpha)
{
    if (actual.empty()) {
        return cv::Mat();
    }

    if (acc.empty()) {
        acc = cv::Mat::zeros(actual.size(), CV_64FC3);
        alpha *= 1;
    }

    cv::accumulateWeighted(actual, acc, alpha);
    return acc.clone() / 255.0;
}

cv::Mat MotionSegmentation::substractBack(cv::Mat actual, cv::Mat back)
{
    if (actual.empty() || back.empty()) {
        return cv::Mat();
    }

    if (back.type() != CV_64FC3 && back.type() != CV_32FC3) {
        return cv::Mat();
    }

    cv::Mat conv, diff;
    back.convertTo(conv, CV_8UC3, 255);
    cv::absdiff(actual, conv, diff);

    return diff.clone();

}

void MotionSegmentation::resetDiffHistory()
{
    diff_history.clear();
    diff_history = std::vector<cv::Mat>(diff_history_size);
}

void MotionSegmentation::resetMotionBuffer()
{
    motion_buffer.clear();
    motion_buffer = std::vector<cv::Mat>(motion_buffer_size);
}
