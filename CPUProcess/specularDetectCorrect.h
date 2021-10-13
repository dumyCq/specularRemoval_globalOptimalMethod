#ifndef SPECULAR_DETECT_CORRECT_H
#define SPECULAR_DETECT_CORRECT_H

#include <opencv2\opencv.hpp>

class specularDetectCorrect {

public:
	specularDetectCorrect();
	~specularDetectCorrect();

	void initialize(int imageRows, int imageCols);

	// return final selected svgSelectImage
	cv::Mat detect(cv::Mat image);

	// return final hsv corrected image in rgb
	cv::Mat correct(cv::Mat image, cv::Mat mask);

private:

	// parameter for the algorithm
	float alpha_1;
	float alpha_2;
	float alpha;

	double tao;
	double sigma;
	double sigma_pow;

	int windowsize_half;

	// variable for the algorithm
	cv::Mat minimumImage, hsvSpace, gradientImage, svSelectImage, gSelectImage, svgSelectImage, svgWeightImage, recoverImage;
	cv::Mat showImage;
};


#endif

