//
// Created by mazuzel on 04/06/2020.
//

#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>

#include <stdio.h>

using namespace cv;

int main() {

    // Create model
    Model m("../graph_frz.pb");

    // Create Tensors
    Tensor input(m, "generator/generator_inputs");
    Tensor prediction(m, "generator/generator_outputs");

    // Read image
    int imgRows = 256;
    int imgCols = 256;

    cv::Mat img, inp, scaled;

    // Read image
    img = cv::imread("../images/test_image.png", cv::IMREAD_COLOR);

    cv::resize(img, inp, cv::Size(imgCols, imgRows));
    cv::cvtColor(img, inp, cv::COLOR_BGR2RGB);
	
    std::vector<float> img_data;

    // Scale image to range [0, 1]
    for(int y=0;y<img.rows;y++){
      for(int x=0;x<img.cols;x++){
	Vec3b & color = inp.at<Vec3b>(y,x); //not optimal, to be improved
	for(int channel=0; channel<3; channel++){
	  img_data.push_back((float)(color[channel]/255.0));
	}
      }
    }

    input.set_data(img_data, {1, imgCols, imgRows, 3});

    // Run and show predictions
    m.run(input, prediction);

    // Get tensor with output image
    std::vector<float> imgVector = prediction.Tensor::get_data<float>();

    unsigned char* pixels = (unsigned char*)malloc(sizeof(unsigned char) * imgRows*imgCols*3);
    for (int j = 0; j < imgVector.size(); j++){
      pixels[j] = (unsigned char)((imgVector[j]+1.0)*127.5);
    }
    cv::Mat image(imgRows, imgCols, CV_8UC3, pixels);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	
    imwrite("output.jpg", image);
}
