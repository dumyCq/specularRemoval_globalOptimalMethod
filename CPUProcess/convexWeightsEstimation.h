#ifndef CONVEX_WEIGHTS_ESTIMATION_H
#define CONVEX_WEIGHTS_ESTIMATION_H

#include <opencv2\opencv.hpp>

class convexWeightsEstimation {

public:
	convexWeightsEstimation();
	~convexWeightsEstimation();

	void initialize(int imageRows, int imageCols);

	// projection operator
	void oneDProjection(double& weight);
	void twoDProjection(double& weight);

	cv::Mat computeWeights(cv::Mat image, cv::Mat specularFree, cv::Mat highlightRegion);
	cv::Mat getDiffuseImage(cv::Mat image, cv::Mat mask);

private:

	double h_k;	// step_size
	double beta_1;	// weights for diffuse grandient regularization term
	double beta_2;	// weights for specular grandient regularization term


	cv::Mat M_d, M_s, specularComponent, diffuseImage, p_x, p_y, q_x, q_y;

};

#endif