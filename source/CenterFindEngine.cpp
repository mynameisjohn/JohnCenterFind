#include "CenterFindEngine.h"
#include <iostream>

static void RecenterImage(cv::UMat& img, float m = 0.f, float M = 100.f) {

}

static void ShowImage(cv::UMat& img) {

}

CenterFindEngine::Parameters(std::array<std::string, 12> args){

}

CenterFindEngine::Data(std::string fileName) {

}

CenterFindEngine::BandPass(int radius, float hwhm) {

}

void CenterFindEngine::BandPass::operator()(cv::UMat& img) {

}

CenterFindEngine::LocalMax(int radius, float pctl_thresh) {

}

void CenterFindEngine::LocalMax::operator()(cv::UMat& img) {

}

CenterFindEngine::Statistics(int mask_radius, int feature_radius) {

}

CenterFindEngine::PMetricsVec CenterFindEngine::Statistics::operator()(cv::UMat& img) {

}

CenterFindEngine::CenterFindEngine(CenterFindEngine::Parameters params) {

}

//
//#include <opencv2/gpu/gpu.hpp>
//
//void CenterFindEngine::showImage(Mat& img){
//	Mat x = img;
//
//	//RecenterImage(x, 1);
//	double min(1), max(2);
//	minMaxLoc(x, &min, &max);
//	x = (x - min) / (max - min);
//
//	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	imshow("Display window", x);                   // Show our image inside it.
//	waitKey(0);
//}
//
//void CenterFindEngine::RecenterImage(Image& img, double range){
//	double min(1), max(2);
//	minMaxLoc(img, &min, &max);
//	subtract(img, min, img);
//
//	//cout << min << "\n" << max << endl;
//#ifdef OCL_OCV
//	multiply(range / (max - min), img, img);
//#elif defined CU_OCV
//	multiply(img,range / (max - min),img);
//#else
//	img *= (range / (max - min));
//#endif
//}
//
//CenterFindEngine::Parameters::Parameters(string params[12]){
//	int idx(0);
//
//	infile_stem = params[idx++];
//	outfile_stem = params[idx++];
//	file_extension = ".tif";
//
//	sstrm(params[idx++]) >> start_frameofstack;
//	sstrm(params[idx++]) >> end_frameofstack;
//	sstrm(params[idx++]) >> start_stack;
//	sstrm(params[idx++]) >> end_stack;
//
//	sstrm(params[idx++]) >> feature_radius;
//	sstrm(params[idx++]) >> hwhm_length;
//	sstrm(params[idx++]) >> dilation_radius;
//	sstrm(params[idx++]) >> mask_radius;
//	sstrm(params[idx++]) >> pctle_threshold;
//	sstrm(params[idx]) >> testmode;
//
//}
//
//CenterFindEngine::BandPassEngine::BandPassEngine(int radius, float h)
//	: m_Radius(radius){
//	int diameter = 2 * m_Radius + 1;
//
//#if defined OCL_OCV || defined CU_OCV
//	//Mat h_Gaussian = getGaussianKernel(diameter, (h / 0.8325546) / 2, CV_32F);
//	Mat h_Circle = Mat::zeros({ diameter, diameter }, CV_32F);
//	circle(h_Circle, { m_Radius, m_Radius }, m_Radius, 1.f, -1);
//
//	Gaussian = createGaussianFilter_GPU(CV_32F, { diameter, diameter }, (h / 0.8325546) / 2);
//	Circle = createLinearFilter_GPU(CV_32F, CV_32F, h_Circle);
//#else
//	Gaussian = createGaussianFilter(CV_32F, { diameter, diameter }, (h / 0.8325546) / 2);
//	//Gaussian = getGaussianKernel(diameter, (h / 0.8325546) / 2, CV_32F);
//	Mat tmp_Circle = Mat::zeros({ diameter, diameter }, CV_32F);
//	circle(tmp_Circle, { m_Radius, m_Radius }, m_Radius, 1.f, -1);
//	Circle = createLinearFilter(CV_32F, CV_32F, tmp_Circle);
//
//	//Gaussian = Image(h_Gaussian);
//	//Circle = Image(h_Circle);
//#endif
//
//}
//
//void CenterFindEngine::BandPassEngine::operator() (CenterFindData& img){
//	//sepFilter2D(img.in, img.bpass, -1, Gaussian, Gaussian);
//	//filter2D(img.in, img.tmp, -1, Circle);
//	Gaussian->apply(img.in, img.bpass);
//	Circle->apply(img.in,img.tmp);
//
//#ifdef OCL_OCV
//	multiply(1 / (3 * pow(m_Radius, 2)), img.tmp, img.tmp);
//#elif defined CU_OCV
//	divide(img.tmp, 3 * pow(m_Radius, 2), img.tmp);
//#else
//	img.tmp = img.tmp / (3 * pow(m_Radius, 2));
//#endif
//	subtract(img.bpass, img.tmp, img.bpass);
//	threshold(img.bpass, img.bpass, 0, 1, THRESH_TOZERO);
//}
//
//CenterFindEngine::LocalMaxEngine::LocalMaxEngine(int radius, float pctl_thresh)
//	: m_Radius(radius), m_Pctl_Threshold(pctl_thresh){
//	int diameter = 2 * m_Radius + 1;
//
//	//Works with OCL?
//	Mat tmp_Dilation = getStructuringElement(MORPH_ELLIPSE, { diameter, diameter });
//#if defined OCL_OCV || defined CU_OCV
//	Dilation = createMorphologyFilter_GPU(MORPH_DILATE, CV_8U, tmp_Dilation);
//#else
//	Dilation = createMorphologyFilter(MORPH_DILATE, CV_32F, tmp_Dilation);
//#endif
//}
//
//void CenterFindEngine::LocalMaxEngine::operator()(CenterFindData& img){
//	const float epsilon(0.0000001f);
//
//	//img.bpass_thresh = m_Pctl_Threshold;
//	img.bpass_thresh.setTo(m_Pctl_Threshold);
//	RecenterImage(img.bpass);
//	max(img.bpass, Threshold, img.bpass_thresh);
//	
////Both the CUDA and OCL OpenCV docs say this is only compatible with 
////byte based image formats. I'm getting away with it on OCL somehow...
//#ifdef CU_OCV
////	multiply(img.bpass_thresh,255.f,img.bpass_thresh);
///*
//	img.bpass_thresh.convertTo(img.bpass_thresh,CV_8U);
//	img.tmp.convertTo(img.tmp,CV_8U);
//
//	Dilation->apply(img.bpass_thresh, img.tmp);
//
//	img.bpass_thresh.convertTo(img.bpass_thresh,CV_32F);
//	img.tmp.convertTo(img.tmp,CV_32F);
//*/
////	divide(img.bpass_thresh,255.f,img.bpass_thresh);
////	divide(img.tmp,255.f,img.tmp);
//#else
//	Dilation->apply(img.bpass_thresh, img.tmp);
//#endif
//	//dilate(img.bpass_thresh, img.tmp, Dilation);
//	subtract(img.bpass, img.tmp, img.local_max);
//	exp(img.local_max, img.local_max);
//	threshold(img.local_max, img.local_max, 1 - epsilon, 1, THRESH_BINARY);
//
//	//img.local_max.convertTo(img.local_max, CV_8U);
//	/*
//
//		exp(img.local_max, img.local_max);
//		double m(0), M(0);
//		minMaxLoc(img.local_max, &m, &M);
//
//		threshold(img.local_max, img.local_max, 1 - epsilon, 1, THRESH_BINARY);*/
//}
//
//CenterFindEngine::StatisticsEngine::StatisticsEngine(int mask_radius, int feature_radius)
//	: m_Mask_Radius(mask_radius), m_Feature_Radius(feature_radius){
//	int diameter = 2 * m_Mask_Radius + 1;
//
//	//cv::Mat_<float> circTmp = cv::Mat_<float>(diameter, diameter, 0.f),
//	//	rxTmp = cv::Mat_<float>(diameter, diameter, 0.f),
//	//	ryTmp = cv::Mat_<float>(diameter, diameter, 0.f),
//	//	r2Tmp = cv::Mat_<float>(diameter, diameter, 0.f);
//
//	Circle = rX = rY = r2 = cv::Mat_<float>(diameter, diameter, 0.f);
//
//	circle(Circle, { m_Mask_Radius, m_Mask_Radius }, m_Mask_Radius, 1.f, -1);
//
//	for (int i = 0; i < diameter; i++)
//		for (int j = 0; j < diameter; j++){
//			rX.at<float>(i, j) = float(j + 1);
//			r2.at<float>(i, j) += float(pow(j - m_Mask_Radius, 2));
//		}
//
//	for (int i = 0; i < diameter; i++)
//		for (int j = 0; j < diameter; j++){
//			rY.at<float>(i, j) = float(i + 1);
//			r2.at<float>(i, j) += float(pow(i - m_Mask_Radius, 2));
//		}
//
//	threshold(r2, r2, pow(mask_radius, 2), 1, THRESH_TOZERO_INV);
//	multiply(rX, Circle, rX);
//	multiply(rY, Circle, rY);
//
//	////Change the constructor?
//	//Circle = Image(circTmp);
//	//rX = Image(rxTmp);
//	//rY = Image(ryTmp);
//	//r2 = Image(r2Tmp);
//
////#ifdef CU_OCV
////	Circle.upload(circTmp);
////	rX.upload(rxTmp);
////	rY.upload(ryTmp);
////	r2.upload(r2Tmp);
////#else
////
////	//When OCL is on, these do the upload
////	Circle = circTmp;
////	rX = rxTmp;
////	rY = ryTmp;
////	r2 = r2Tmp;
////#endif
//}
//
//vector<CenterFindEngine::ParticleData> CenterFindEngine::StatisticsEngine::operator()(CenterFindData& img){
//	const float epsilon(0.0001f);
//
//	vector<ParticleData> ret;
//
//	int counter(0);
//	int border = m_Feature_Radius;
//	int minx = border, miny = border;
//	int maxx = img.cols() - minx;
//	int maxy = img.rows() - minx;
//	int diameter = m_Mask_Radius * 2 + 1;
//
////#if !(defined OCL_OCV || defined CU_OCV)
//	for (int i = 0; i < img.rows()*img.cols(); i++){
//		auto * ptr = img.particles.ptr<unsigned char>();
//		if (ptr[i]/*fabs(ptr[i] - 1) < epsilon*/){
//			int xval(i%img.cols());
//			int yval(floor((float)i / img.cols()));
//			if (xval > minx && xval < maxx && yval > miny && yval < maxy) {
//				int mask = m_Mask_Radius;
//				Rect extract(xval - mask, yval - mask, diameter, diameter);
//				Mat e_square(img.input(extract)), result;
//				multiply(e_square, Circle, result);
//				float total_mass(sum(result)[0]);
//
//				if (total_mass > 0) {
//					multiply(e_square, rX, result);
//					float x_offset = ((sum(result)[0]) / total_mass) - mask - 1;
//
//					multiply(e_square, rY, result);
//					float y_offset = (sum(result)[0] / total_mass) - mask - 1;
//
//					multiply(e_square, r2, result);
//					float r2_val = (sum(result)[0] / total_mass);
//
//					Mat m_square = img.particles(extract);
//					float multiplicity(sum(m_square)[0]);
//
//					ParticleData p = {
//						float(i),
//						xval + x_offset,
//						yval + y_offset,
//						x_offset,
//						y_offset,
//						total_mass,
//						r2_val,
//						multiplicity
//					};
//					ret.push_back(p);
//
//					counter++;
//				}
//			}
//		}
//	}
////#endif
////	cout << counter << endl;
//	return ret;
//}
//#include <chrono>
//#include <thread>
//
////Resource Aquisition Is Initialization?
//CenterFindEngine::CenterFindEngine(string params[12])
//	: m_Params(params),
//	m_BandPass(m_Params.feature_radius, m_Params.hwhm_length),
//	m_LocalMax(m_Params.dilation_radius, m_Params.pctle_threshold),
//	m_Statistics(m_Params.mask_radius, m_Params.feature_radius)
//{
//	auto TIFF_to_OCV = [](FIBITMAP * dib){
//		Mat image = Mat::zeros(FreeImage_GetWidth(dib), FreeImage_GetHeight(dib), CV_8UC3);
//		Image ret;
//
//		FreeImage_ConvertToRawBits(image.data, dib, image.step, 24, 0xFF, 0xFF, 0xFF, true);
//
//		cvtColor(image, image, CV_RGB2GRAY);
//
//		image.convertTo(image, CV_32FC1);
//
//		//ret = Image(image);
//
//		return image; //ret;
//	};
//	for (int i = m_Params.start_stack; i < m_Params.end_stack; i++){
//		string fileName = m_Params.getFileName(i).c_str();
//
//		FIMULTIBITMAP * FI_input =
//			FreeImage_OpenMultiBitmap(FIF_TIFF, m_Params.getFileName(i).c_str(),
//			FALSE, TRUE, TRUE, TIFF_DEFAULT);
//
//		for (int j = m_Params.start_frameofstack; j < m_Params.end_frameofstack; j++)
//			m_Images.emplace_back(TIFF_to_OCV(FreeImage_LockPage(FI_input, j - 1)));
//
//		FreeImage_CloseMultiBitmap(FI_input, TIFF_DEFAULT);
//	}
//
//
//	m_LocalMax.Threshold = Image({ m_Images.front().rows(), m_Images.front().cols() }, CV_32F, m_Params.pctle_threshold);
//
//	thread T([&](){
//		int nProcessed(0);
//		for (auto& img : m_Images){
//			unique_lock<mutex> lk(img.m);
//			img.cv.wait(lk, [&img]{return img.goodToGo; });
//			img.m_Data = m_Statistics(img);
//			lk.unlock();
//			//cout << "Statistics gathered from " << nProcessed++ << " images" << endl;
//		}
//	});
//
//	auto begin = chrono::steady_clock::now();
//	int nProcessed(0);
//	for (auto& img : m_Images){
//
//		Mat display = Mat(img.in);
////		img.in.download(display);
//		showImage(display);
//
//		RecenterImage(img.in);
//
//		m_BandPass(img);
//
//		display = Mat(img.bpass);/*.download(display);*/
//		showImage(display);
//
//		m_LocalMax(img);
////		img.local_max.download(display);
//		display = Mat(img.local_max);
//		showImage(display);
//
//		img.local_max.convertTo(img.local_max, CV_8U);
//#ifndef CU_OCV
//		img.particles = img.local_max;
//#else
//		img.local_max.download(img.particles);
//#endif
//		RecenterImage(img.bpass_thresh);
//		{
//			lock_guard<mutex> lk(img.m);
//			img.goodToGo = true;
//		}
//		img.cv.notify_one();
//		//cout << "Filters run on " << nProcessed++ << " images" << endl;
//		//m_Data.push_back(m_Statistics(img));
//	}
//	T.join();
//	auto end = chrono::steady_clock::now();
//
//	string msg;
//
//#ifdef OCL_OCV
//	msg = "OpenCL time took ";
//#elif defined CU_OCV
//	msg = "CUDA time took ";
//#else
//	msg = "Host time took ";
//#endif
//
//	cout << msg << chrono::duration<double, milli>(end-begin).count() << endl;
//}
