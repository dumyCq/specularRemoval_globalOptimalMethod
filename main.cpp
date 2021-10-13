#include <opencv2\opencv.hpp>
#include "CPUProcess\specularDetectCorrect.h"
#include "CPUProcess\convexWeightsEstimation.h"

int main(int argc, char **argv) 
{

	if(argc != 2) 
	{
		printf("Usage: GlobalOptimalMethodSpecularRemove.exe imagefile.extension\n");
		return -1;
	}

	cv::Mat inputImage = cv::imread(argv[1]);
	cv::Mat selectImage, noSpecularImage, diffuseMask, diffuseImage;

	// detect specular region and correct in HSV space
	specularDetectCorrect SpecularDetectCorrectInstance;
	SpecularDetectCorrectInstance.initialize(inputImage.rows, inputImage.cols);
	selectImage = SpecularDetectCorrectInstance.detect(inputImage);

	// show the specular region
	cv::imshow("specularRegion", selectImage);
	cv::waitKey();

	// note that this SpecularImage is 32FC3
	noSpecularImage = SpecularDetectCorrectInstance.correct(inputImage, selectImage);

	// show the hue and saturation corrected image and highlight on valur
	cv::imshow("specularFree image", noSpecularImage);
	cv::waitKey();

	// estimate w_d and return diffuse image
	convexWeightsEstimation ConvexWeightsEstimationInstance;
	ConvexWeightsEstimationInstance.initialize(inputImage.rows, inputImage.cols);
	diffuseMask = ConvexWeightsEstimationInstance.computeWeights(inputImage, noSpecularImage, selectImage);
	//diffuseImage = ConvexWeightsEstimationInstance.getDiffuseImage(noSpecularImage, diffuseMask);
	
	//// show the final result
	//cv::imshow("diffuse mask", diffuseMask);
	//cv::imwrite("diffuseMask.jpg", diffuseMask);
	//cv::waitKey();

	//cv::imshow("diffuse image", diffuseImage);
	//cv::imwrite("diffuseImage.jpg", diffuseImage);
	//cv::waitKey();

	return 0;
	
}
