#include "specularDetectCorrect.h"

#include <math.h>
#include <highgui.h>

specularDetectCorrect::specularDetectCorrect()
{
	// alpha and tao are different under medical and natural 

	alpha = 0.1;
	alpha_1 = 0.33;
	alpha_2 = 0.67;

	tao = 0.13;
	sigma = 0.01;
	windowsize_half = 3;

	sigma_pow = pow(sigma, 2);
}

specularDetectCorrect::~specularDetectCorrect() {

	// need to delay some heap information
}

void specularDetectCorrect::initialize(int imageRows, int imageCols) {
	minimumImage = cv::Mat(imageRows, imageCols, CV_32FC1);
	hsvSpace = cv::Mat(imageRows, imageCols, CV_32FC3);
	gradientImage = cv::Mat::zeros(imageRows, imageCols, CV_64FC1); 
	svSelectImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	gSelectImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	svgSelectImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	svgWeightImage = cv::Mat::ones(imageRows, imageCols, CV_32FC1);
	recoverImage = cv::Mat(imageRows, imageCols, CV_8UC3);
	//showImage = cv::Mat(imageRows, imageCols, CV_8UC1);
}

cv::Mat specularDetectCorrect::detect(cv::Mat image) {
	
	// get hsvSpace from input image
	cv::Mat image_float = cv::Mat(image.rows, image.cols, CV_32FC3);
	image.convertTo(image_float, CV_32FC3, 1.f / 255.f);
	cv::cvtColor(image_float, hsvSpace, CV_BGR2HSV_FULL);

	// get svSelectImage from HSV space
	// @qxc62 09/29 low precise, 01/13 get the 32 bit float high precise
	for (int pixel = 0; pixel < hsvSpace.rows * hsvSpace.cols; pixel++) {
		float saturation = hsvSpace.ptr<float>()[pixel * 3 + 1];
		float value = hsvSpace.ptr<float>()[pixel * 3 + 2];

		svSelectImage.ptr<unsigned char>()[pixel] = (((saturation < alpha_1) && (value > alpha_2)) ? 255 : 0);

	}

	//cv::imshow("svSelect Image", svSelectImage);
	//cv::waitKey();

	// get minimum image from original input image
	for (int pixel = 0; pixel < image.rows * image.cols; pixel++) {
		float red = image_float.ptr<float>()[pixel * 3 + 2];
		float green = image_float.ptr<float>()[pixel * 3 + 1];
		float blue = image_float.ptr<float>()[pixel * 3 + 0];
		// compare each value and get output minimum image
		minimumImage.ptr<float>()[pixel] = std::min(red, std::min(green, blue));
	}

	// calculate gradient image, gradient select and union
	for (int r = 1; r < minimumImage.rows - 1; r++) {
		for (int c = 1; c < minimumImage.cols - 1; c++) {
			// calculate gradient image from minimum Image
			double cal_temp = std::pow((minimumImage.at<float>(r + 1, c) - minimumImage.at<float>(r - 1, c)), 2) + std::pow((minimumImage.at<float>(r, c + 1) - minimumImage.at<float>(r, c - 1)), 2);
			gradientImage.at<double>(r, c) = std::sqrt(cal_temp) / 2;
			// select pixel in gradient image
			gSelectImage.at<unsigned char>(r, c) = (gradientImage.at<double>(r, c) > tao) ? 255 : 0;
			// select pixel union with sv selected image
			svgSelectImage.at<unsigned char>(r, c) = (gSelectImage.at<unsigned char>(r, c) > 250) ? (gSelectImage.at<unsigned char>(r, c)) : (svSelectImage.at<unsigned char>(r, c));
		}
	}

	return svgSelectImage;
}

cv::Mat specularDetectCorrect::correct(cv::Mat image, cv::Mat mask) {

	// get hsvSpace from input image
	cv::Mat image_float = cv::Mat(image.rows, image.cols, CV_32FC3);
	image.convertTo(image_float, CV_32FC3, 1.0 / 255.0);
	// note that Hue 0-360, Saturation 0-1, Value 0-1 for 32bit input transform
	cv::cvtColor(image_float, hsvSpace, CV_BGR2HSV_FULL);

	// normalize hue to 0-1
	for (int pixel = 0; pixel < hsvSpace.rows * hsvSpace.cols; pixel++) {
		float hue = hsvSpace.ptr<float>()[pixel * 3 + 0];
		hsvSpace.ptr<float>()[pixel * 3 + 0] = hue / 360.f;
	}

	//// check the pre-process HSV space
	//cv::Mat dst;
	//dst.create(hsvSpace.size(), hsvSpace.depth());

	//int ch[] = { 0, 0 };
	//cv:mixChannels(&hsvSpace, 1, &dst, 1, ch, 1);
	//cv::imshow("Original H channel", dst);

	//int ch1[] = { 1, 0 };
	//cv::mixChannels(&hsvSpace, 1, &dst, 1, ch1, 1);
	//cv::imshow("Original S channel", dst);

	//int ch2[] = { 2, 0 };
	//cv::mixChannels(&hsvSpace, 1, &dst, 1, ch2, 1);
	//cv::imshow("Original V channel", dst);
	//cv::waitKey();

	// calculate svg weight here, function (13)
	for (int pixel = 0; pixel < svgWeightImage.rows * svgWeightImage.cols; pixel++) {
		if (mask.ptr<unsigned char>()[pixel] > 250) {
			svgWeightImage.ptr<float>()[pixel] = alpha;
		}
		else {
			svgWeightImage.ptr<float>()[pixel] = 1.f;
		}
	}

	// correct surrounding pixels until finished all
	bool continue_flag = true;

	cv::Mat mask_erode;
	cv::Mat mask_large = mask.clone();

	cv::Mat Element_erode = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2, 2));
	cv::erode(mask_large, mask_erode, Element_erode);

	// process loop to correct
	while (continue_flag) {
		// correct hue and saturation
		//cv::Mat ProcessImage = cv::Mat::zeros(mask.rows, mask.cols, CV_32FC1);
		for (int r = windowsize_half; r < hsvSpace.rows - windowsize_half; r++) {
			for (int c = windowsize_half; c < hsvSpace.cols - windowsize_half; c++) {
				double h_update = 0.0;
				double s_update = 0.0;
				double svg_weight_sum = 0.0;
				double hs_weight_sum = 0.0;
				if (mask_large.at<unsigned char>(r, c) > 250 && mask_erode.at<unsigned char>(r, c) < 250) {
					for (int wr = -windowsize_half; wr <= windowsize_half; wr++) {
						for (int wc = -windowsize_half; wc <= windowsize_half; wc++) {

							double h_temp = hsvSpace.at<cv::Vec3f>(r + wr, c + wc)[0];
							double s_temp = hsvSpace.at<cv::Vec3f>(r + wr, c + wc)[1];
							double h_current = hsvSpace.at<cv::Vec3f>(r , c )[0];

							// function (11)
							h_update += h_temp * svgWeightImage.at<float>(r + wr, c + wc);
							svg_weight_sum += svgWeightImage.at<float>(r + wr, c + wc);

							// function (12) and (14)
							s_update += s_temp * exp(-pow((1 - s_temp), 2)) * exp(-pow((h_current - h_temp), 2) / sigma_pow);
							hs_weight_sum += exp(-pow((1 - s_temp), 2)) * exp(-pow((h_current - h_temp), 2) / sigma_pow);

						}
					}

					// finalize function (11) and (12)
					hsvSpace.at<cv::Vec3f>(r, c)[0] = h_update / svg_weight_sum;
					hsvSpace.at<cv::Vec3f>(r, c)[1] = s_update / hs_weight_sum;
					hsvSpace.at<cv::Vec3f>(r, c)[2] = 1.f;
					//svgWeightImage.at<float>(r, c) = 1.f; // this will blur the inpainting a lot
				}

			}
		}

		mask_large = mask_erode.clone();
		cv::erode(mask_large, mask_erode, Element_erode);
	
		continue_flag = (cv::countNonZero(mask_erode) > 1);
	}

	//// check the post-process HSV space
	//cv::mixChannels(&hsvSpace, 1, &dst, 1, ch, 1);
	//cv::imshow("H channel", dst);


	//cv::mixChannels(&hsvSpace, 1, &dst, 1, ch1, 1);
	//cv::imshow("S channel", dst);

	//cv::mixChannels(&hsvSpace, 1, &dst, 1, ch2, 1);
	//cv::imshow("V channel", dst);
	//cv::waitKey();

	// rescale hue to 0-360
	for (int pixel = 0; pixel < hsvSpace.rows * hsvSpace.cols; pixel++) {
		float hue = hsvSpace.ptr<float>()[pixel * 3 + 0];
		hsvSpace.ptr<float>()[pixel * 3 + 0] = hue * 360.f;
	}

	// get recover image after correct hue and saturation
	cv::cvtColor(hsvSpace, recoverImage, CV_HSV2BGR_FULL);

	return recoverImage;
}