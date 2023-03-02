#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


class PredictionProcessor
{
private:
    cv::Scalar lowerBound;
    cv::Scalar upperBound;


public:
    // Default constructor
    PredictionProcessor();

    // Constructor with custom HSV bounds
    PredictionProcessor(const cv::Scalar &lowerBound, const cv::Scalar &upperBound);
    ~PredictionProcessor();

    // Compute the centers of the largest green objects in the image
    void computeCenters(const cv::Mat &image, std::vector<cv::Point2i> &dest);



};


