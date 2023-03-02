#include "../inc/PredictionProcessor.hpp"

PredictionProcessor::PredictionProcessor()
{
    // Default values for HSV to filter green color
    // Look up hue ranges to see color ranges
    this->lowerBound = cv::Scalar(30, 30, 30);
    this->upperBound = cv::Scalar(90, 255, 255);
}

PredictionProcessor::PredictionProcessor(const cv::Scalar &lowerBound, const cv::Scalar &upperBound) 
 : lowerBound(lowerBound), upperBound(upperBound){}

PredictionProcessor::~PredictionProcessor(){}


void PredictionProcessor::computeCenters(const cv::Mat &image, std::vector<cv::Point2i> &dest)
{
    // Convert image to HSV because it is easier to filter colors in the HSV color-space.
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Filter out green pixels
    cv::Mat greenOnly;
    cv::inRange(hsv, this->lowerBound, this->upperBound, greenOnly);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(greenOnly, labels, stats, centroids);

    // Sort centroids by size
    std::vector<std::pair<int, cv::Point2i>> sizeCentroids; // (size, centroid)
    for (int i = 0; i < stats.rows; i++)
    {
        int size = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Point2i centroidInt = cv::Point2i(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        sizeCentroids.push_back(std::make_pair(size, centroidInt));
        // std::cout << "(size, centroid): (" << size << ", " << centroidInt << ")" << std::endl;
    }
    std::sort(sizeCentroids.begin(), sizeCentroids.end(), [](const std::pair<int, cv::Point2i> &a, const std::pair<int, cv::Point2i> &b) {
        return a.first > b.first;
    });

    // Extract centroids from sizeCentroids into a vector
    dest.resize(sizeCentroids.size());
    for (int i = 0; i < dest.size(); i++)
    {
        dest[i] = sizeCentroids[i].second;
        // std::cout << "sorted: " << sizeCentroids[i].second << std::endl;
    }
}
