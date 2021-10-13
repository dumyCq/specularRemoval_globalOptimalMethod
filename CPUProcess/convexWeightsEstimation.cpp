#include "convexWeightsEstimation.h"

#include <vector>
#include <math.h>

convexWeightsEstimation::convexWeightsEstimation() {
	h_k = 2e-5;
	beta_1 = 1;
	beta_2 = 0.1;
}

convexWeightsEstimation::~convexWeightsEstimation() {
	// need to delay some heap information
}

void convexWeightsEstimation::initialize(int imageRows, int imageCols) {

	M_d = cv::Mat::ones(imageRows, imageCols, CV_64FC1);
	M_s = cv::Mat::ones(imageRows, imageCols, CV_64FC1);

	p_x = cv::Mat::zeros(imageRows, imageCols, CV_64FC1);
	p_y = cv::Mat::zeros(imageRows, imageCols, CV_64FC1);
	q_x = cv::Mat::zeros(imageRows, imageCols, CV_64FC1);
	q_y = cv::Mat::zeros(imageRows, imageCols, CV_64FC1);

	specularComponent = cv::Mat(imageRows, imageCols, CV_32FC3);
	diffuseImage = cv::Mat(imageRows, imageCols, CV_32FC3);
}

void convexWeightsEstimation::oneDProjection(double& weight) {
	if (weight > 1) { weight = 1.0; }
	else if (weight < 0) { weight = 0.0; }
	else { weight = weight; }
}

void convexWeightsEstimation::twoDProjection(double& weight) {
	if (weight > 1) { weight = 1.0; }
	else if (weight < -1) { weight = -1.0; }
	else { weight = weight; }
}

cv::Mat convexWeightsEstimation::computeWeights(cv::Mat image, cv::Mat specularFree, cv::Mat highlightRegion) {

	// get the large highlight region
	cv::Mat Element_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 1));
	cv::Mat large_highlight_region;
	cv::morphologyEx(highlightRegion, large_highlight_region, CV_MOP_OPEN, Element_open);

	// find around pixel of highlight region
	cv::Mat Element_dialate = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2));
	cv::Mat large_highlight_region_extend;
	cv::dilate(large_highlight_region, large_highlight_region_extend, Element_dialate);
	
	// use connected component analysis get the label
	cv::Mat reginLabel;
	int num = cv::connectedComponents(large_highlight_region_extend, reginLabel, 8);
	//std::cout << "num:" << num << std::endl;

	// calculate the surrounding average
	std::vector<std::vector<int>> cluster(num, std::vector<int>(2,0));
	for (int pixel = 0; pixel < reginLabel.rows * reginLabel.cols; pixel++) {
		int label = reginLabel.ptr<int>()[pixel];
		if (label > 0) {
			cluster[label][0] += (int)image.ptr<unsigned char>()[pixel * 3 + 0] + (int)image.ptr<unsigned char>()[pixel * 3 + 1] + (int)image.ptr<unsigned char>()[pixel * 3 + 2];
			cluster[label][1] += 1;
		}
	}

	std::vector<double> cluster_average(num,0.0);
	for (int n = 1; n < num; n++) {
		cluster_average[n] = cluster[n][0] / (3 * cluster[n][1]);
		//std::cout << cluster_average[n] << std::endl;
	}

	// calculate gradient method and iterative update
	int count = 0;

	// define specular image
	specularComponent = specularFree.clone();
	
	cv::Mat specularFree_float, specularComponent_float, image_float;
	specularFree.convertTo(specularFree_float, CV_64FC3);
	specularComponent.convertTo(specularComponent_float, CV_64FC3);
	image.convertTo(image_float, CV_64FC3, 1 / 255.0);

	for (int pixel = 0; pixel < large_highlight_region.rows * large_highlight_region.cols; pixel++) {
		int label = reginLabel.ptr<int>()[pixel];
		if (large_highlight_region.ptr<unsigned char>()[pixel] > 250 && label > 0) {
			specularComponent_float.ptr<double>()[pixel * 3 + 2] = 1.0 - cluster_average[label]/255.0;
			//specularComponent_float.ptr<double>()[pixel * 3 + 2] = 1.0;
			specularComponent_float.ptr<double>()[pixel * 3 + 0] = 1.0;
			specularComponent_float.ptr<double>()[pixel * 3 + 1] = 1.0;
		}
		else {
			specularComponent_float.ptr<double>()[pixel * 3 + 2] = 0.0;
			specularComponent_float.ptr<double>()[pixel * 3 + 0] = 0.0;
			specularComponent_float.ptr<double>()[pixel * 3 + 1] = 0.0;
		}

	}

	//cv::imshow("specularComponent", specularComponent_float);
	//cv::waitKey();

	std::cout << std::endl;

	while (count < 100000) {

		// loop to check when is the good ending
		if (count % 500 == 0) {
			cv::Mat M_d_float;
			M_d.convertTo(M_d_float, CV_32FC1);
			cv::Mat showTemp = getDiffuseImage(specularFree, M_d_float);
			if (count == 0) {
				cv::imshow("original input", showTemp);
				cv::waitKey(10);
			}
			else {
				cv::imshow("changing result", showTemp);
				cv::waitKey(10);
			}
			
		}

		cv::Mat div_p_x = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		cv::Mat div_p_y = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		cv::Mat div_q_x = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		cv::Mat div_q_y = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);

		cv::Mat div_m_s_x = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		cv::Mat div_m_s_y = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		cv::Mat div_m_d_x = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		cv::Mat div_m_d_y = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
		
		// precompute div and gradient in x and y direction
		for (int r = 0; r < image.rows - 1; r++) {
			for (int c = 0; c < image.cols - 1; c++) {
				div_q_x.at<double>(r, c) = q_x.at<double>(r, c) - q_x.at<double>(r + 1, c);
				div_q_y.at<double>(r, c) = q_y.at<double>(r, c) - q_y.at<double>(r, c + 1);
				div_p_x.at<double>(r, c) = p_x.at<double>(r, c) - p_x.at<double>(r + 1, c);
				div_p_y.at<double>(r, c) = p_y.at<double>(r, c) - p_y.at<double>(r, c + 1);

				div_m_d_x.at<double>(r, c) = M_d.at<double>(r, c) - M_d.at<double>(r + 1, c);
				div_m_d_y.at<double>(r, c) = M_d.at<double>(r, c) - M_d.at<double>(r, c + 1);
				div_m_s_x.at<double>(r, c) = M_s.at<double>(r, c) - M_s.at<double>(r + 1, c);
				div_m_s_y.at<double>(r, c) = M_s.at<double>(r, c) - M_s.at<double>(r, c + 1);
			}
		}

		// update m_d, m_s, p and q
		for (int pixel = 0; pixel < image.rows * image.cols; pixel++) {

			double specularfree_r = specularFree_float.ptr<double>()[pixel * 3 + 2];
			double specularfree_g = specularFree_float.ptr<double>()[pixel * 3 + 1];
			double specularfree_b = specularFree_float.ptr<double>()[pixel * 3 + 0];

			double specularComponent_r = specularComponent_float.ptr<double>()[pixel * 3 + 2];
			double specularComponent_g = specularComponent_float.ptr<double>()[pixel * 3 + 1];
			double specularComponent_b = specularComponent_float.ptr<double>()[pixel * 3 + 0];

			double image_r = image_float.ptr<double>()[pixel * 3 + 2];
			double image_g = image_float.ptr<double>()[pixel * 3 + 1];
			double image_b = image_float.ptr<double>()[pixel * 3 + 0];

			double m_d = M_d.ptr<double>()[pixel];
			double m_s = M_s.ptr<double>()[pixel];

			double R = m_d * specularfree_r + m_s * specularComponent_r - image_r;
			double G = m_d * specularfree_g + m_s * specularComponent_g - image_g;
			double B = m_d * specularfree_b + m_s * specularComponent_b - image_b;

			//std::cout << R << std::endl;

			// get the parameter and do not need to /255
			double div_p = div_p_x.ptr<double>()[pixel] + div_p_y.ptr<double>()[pixel];
			double div_q = div_q_x.ptr<double>()[pixel] + div_q_y.ptr<double>()[pixel];

			double p_x_element = p_x.ptr<double>()[pixel];
			double p_y_element = p_y.ptr<double>()[pixel];
			double q_x_element = q_x.ptr<double>()[pixel];
			double q_y_element = q_y.ptr<double>()[pixel];

			double div_m_d_x_element = div_m_d_x.ptr<double>()[pixel];
			double div_m_d_y_element = div_m_d_y.ptr<double>()[pixel];
			double div_m_s_x_element = div_m_s_x.ptr<double>()[pixel];
			double div_m_s_y_element = div_m_s_y.ptr<double>()[pixel];


			// calculate update
			double m_d_next = m_d - h_k * (R * specularfree_r + G * specularfree_g + B * specularfree_b - beta_1 * div_q);
			double m_s_next = m_s - h_k * (R * specularComponent_r + G * specularComponent_g + B * specularComponent_b - beta_2 * div_p);
			double p_x_next = p_x_element - h_k * beta_2 * div_m_s_x_element;
			double p_y_next = p_y_element - h_k * beta_2 * div_m_s_y_element;
			double q_x_next = q_x_element - h_k * beta_1 * (div_m_d_x_element - q_x_element);
			double q_y_next = q_y_element - h_k * beta_1 * (div_m_d_y_element - q_y_element);

			// projection
			oneDProjection(m_d_next);
			oneDProjection(m_s_next);
			twoDProjection(p_x_next);
			twoDProjection(p_y_next);

			// update
			M_d.ptr<double>()[pixel] = m_d_next;
			M_s.ptr<double>()[pixel] = m_s_next;
			p_x.ptr<double>()[pixel] = p_x_next;
			p_y.ptr<double>()[pixel] = p_y_next;
			q_x.ptr<double>()[pixel] = q_x_next;
			q_y.ptr<double>()[pixel] = q_y_next;

		}

		count++;
		std::cout << count << " ";
	}

	std::cout << std::endl;

	// return diffuse weight
	return M_d;
}

cv::Mat convexWeightsEstimation::getDiffuseImage(cv::Mat image, cv::Mat mask) {

	for (int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		diffuseImage.ptr<float>()[pixel * 3 + 0] = image.ptr<float>()[pixel * 3 + 0] * mask.ptr<float>()[pixel];
		diffuseImage.ptr<float>()[pixel * 3 + 1] = image.ptr<float>()[pixel * 3 + 1] * mask.ptr<float>()[pixel];
		diffuseImage.ptr<float>()[pixel * 3 + 2] = image.ptr<float>()[pixel * 3 + 2] * mask.ptr<float>()[pixel];
	}
	return diffuseImage;
}