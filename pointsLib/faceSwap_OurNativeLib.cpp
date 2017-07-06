#define SKIP_FRAMES 2
#include "faceSwap_OurNativeLib.h"
#include "facedetect-dll.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <dlib\opencv\cv_image.h>

using namespace std;
using namespace dlib;
#define DETECT_BUFFER_SIZE 0x20000

std::vector<frontal_face_detector> detectors ;

std::vector<shape_predictor> predictors;

JNIEXPORT jintArray JNICALL Java_faceswap_OurNativeLib_getPointsAndRec
(JNIEnv * env, jobject obj, jlong matAddr, jfloat scale, jint min_neighbors,jint minObjectWidth,jint maxObjectWidth,jint dolandmark)
{

	jintArray rezult = nullptr;
	cv::Mat * inputMat = (cv::Mat*)matAddr;
	float dx = 1;// ((float)(*inputMat).cols) / ((float)(*inputMat).cols / 1.2);
	float dy = 1;// ((float)(*inputMat).rows) / ((float)(*inputMat).rows / 1.2);
	/*Mat mat = imread("C:\\Users\\Rinat\\Desktop\\fswp\\008.jpg");
	Mat *inputMat = &mat; (*inputMat) */
	cv::Mat gray;
	//Origin size devide to 4 
	//Size size((*inputMat).cols/1.2, (*inputMat).rows/1.2);
	//resize(*inputMat, gray,size,0.0,0.0,INTER_LANCZOS4);
	cvtColor(*inputMat, gray, CV_BGR2GRAY);
	int * pResults = NULL;
	//pBuffer is used in the detection functions.
	//If you call functions in multiple threads, please create one buffer for each thread!
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		rezult = env->NewIntArray(1);
		return rezult;
	}
	

	///////////////////////////////////////////
	// frontal face detection / 68 landmark detection
	// it's fast, but cannot detect side view faces
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
		scale, min_neighbors, minObjectWidth, maxObjectWidth, dolandmark);


	if ((*pResults) <= 0)
	{
		fprintf(stderr, "Nofaces.\n");
		rezult = env->NewIntArray(1);
		return rezult;
	}

	rezult = env->NewIntArray((*pResults) * 140);
	if (rezult == nullptr)
	{
		fprintf(stderr, "Can not alloc rezult memory.\n");
		rezult = env->NewIntArray(1);
		return rezult;
	}
	jint bodyOfrezult[140];
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		for (int k = 0; k < 4; k+=2)
		{
			//140 всего точек на одно лицо 68+68(кординаты x и y) + 4 точки для квадрата с лицом
			bodyOfrezult[k] = p[k] * dx;
			bodyOfrezult[k+1] = p[k+1] * dy;
		}
		if (dolandmark)
		{
			for (int j = 4; j < 140; j+=2)
			{
				bodyOfrezult[j] = ((int)p[2 + j])*dx;
				bodyOfrezult[j+1] = ((int)p[2 + j+1])*dy;
			}
			env->SetIntArrayRegion(rezult, i * 140, 140 , bodyOfrezult);
		}
	}
	free(pBuffer);
	gray.release();
	return rezult;
}

JNIEXPORT jintArray JNICALL Java_faceswap_OurNativeLib_getPointsAndRecDlib (JNIEnv * env, jobject obj, jlong matAddr, jint detectorIndex, jint shapeIndex,jint SCALE )
{
	try {
		jintArray rezult = nullptr;
		cv::Mat * inputMat = (cv::Mat*)matAddr;
		frontal_face_detector detector = detectors[detectorIndex];
		shape_predictor sp = predictors[shapeIndex];
		cv::Mat im_small;
		cv::Mat im = *(cv::Mat*)matAddr;
		cv::cvtColor(im, im, CV_BGR2GRAY);
		cv::resize(im, im_small, cv::Size(), 1.0 / SCALE, 1.0 / SCALE);
		// Make the image larger so we can detect small faces.
		//pyramid_up(img);
		cv_image<unsigned char> cimg_small(im_small);
		cv_image<unsigned char> cimg(im);

		// Now tell the face detector to give us a list of bounding boxes
		// around all the faces in the image.
		std::vector<rectangle> dets = detector(cimg_small);
		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		rezult = env->NewIntArray(dets.size() * 140);
		jint bodyOfrezult[140];
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			// Resize obtained rectangle for full resolution image. 
			rectangle r(
				(long)(dets[j].left() * SCALE),
				(long)(dets[j].top() * SCALE),
				(long)(dets[j].right() * SCALE),
				(long)(dets[j].bottom() * SCALE)
			);
			full_object_detection shape = sp(cimg, r);
			bodyOfrezult[0] = r.left();
			bodyOfrezult[1] = r.top();
			bodyOfrezult[2] = r.width();
			bodyOfrezult[3] = r.height();
			for (int k = 4, l = 0; k < 140; k += 2, l++)
			{
				dlib::point p = shape.part(l);
				
				bodyOfrezult[k] = p.x();
				bodyOfrezult[k + 1] = p.y();
			}
			env->SetIntArrayRegion(rezult, j * 140, 140, bodyOfrezult);
		}
		if (dets.size() == 0)
		{
			rezult = env->NewIntArray(1);
		}
		im.release();
		return rezult;
	}
	catch (exception& e)
	{
		cerr << e.what();
		return 0;
	}
	return 0;
}

JNIEXPORT jint JNICALL Java_faceswap_OurNativeLib_createDetector(JNIEnv * env, jobject obj)
{
	detectors.push_back(get_frontal_face_detector());
	return detectors.size() - 1;
}

JNIEXPORT jint JNICALL Java_faceswap_OurNativeLib_createShapePredictor(JNIEnv * env, jobject obj)
{
	shape_predictor sp;
	deserialize("C:\\Users\\Rinat\\Desktop\\Photo\\shape_predictor_68_face_landmarks.dat") >> sp;
	predictors.push_back(sp);
	return predictors.size()-1;
}

JNIEXPORT void JNICALL Java_faceswap_OurNativeLib_specifiyHistogram(JNIEnv * env, jobject obj, jlong matAdrSrc, jlong matAdrmask, jlong matadrDst)
{
	cv::Mat * source_image = (cv::Mat*)matAdrSrc;
	cv::Mat * target_image = (cv::Mat*)matadrDst;
	cv::Mat * mask = (cv::Mat*)matAdrmask;
	
	std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
	std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);

	for (size_t i = 0; i < mask->rows; i++)
	{
		auto current_mask_pixel = mask->row(i).data;
		auto current_source_pixel = source_image->row(i).data;
		auto current_target_pixel = target_image->row(i).data;

		for (size_t j = 0; j < mask->cols; j++)
		{
			if (*current_mask_pixel != 0) {
				source_hist_int[0][*current_source_pixel]++;
				source_hist_int[1][*(current_source_pixel + 1)]++;
				source_hist_int[2][*(current_source_pixel + 2)]++;

				target_hist_int[0][*current_target_pixel]++;
				target_hist_int[1][*(current_target_pixel + 1)]++;
				target_hist_int[2][*(current_target_pixel + 2)]++;
			}

			// Advance to next pixel
			current_source_pixel += 3;
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}

	// Calc CDF
	for (size_t i = 1; i < 256; i++)
	{
		source_hist_int[0][i] += source_hist_int[0][i - 1];
		source_hist_int[1][i] += source_hist_int[1][i - 1];
		source_hist_int[2][i] += source_hist_int[2][i - 1];

		target_hist_int[0][i] += target_hist_int[0][i - 1];
		target_hist_int[1][i] += target_hist_int[1][i - 1];
		target_hist_int[2][i] += target_hist_int[2][i - 1];
	}

	// Normalize CDF
	for (size_t i = 0; i < 256; i++)
	{
		source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);
		source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
		source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

		target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
		target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
		target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
	}

	// Create lookup table

	auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
	{
		uint8_t l = 0, r = 255, m;
		while (l < r)
		{
			m = (l + r) / 2;
			if (needle > haystack[m])
				l = m + 1;
			else
				r = m - 1;
		}
		// TODO check closest value
		return m;
	};

	for (size_t i = 0; i < 256; i++)
	{
		LUL[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
		LUL[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
		LUL[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
	}

	// repaint pixels
	for (size_t i = 0; i < mask->rows; i++)
	{
		auto current_mask_pixel = mask->row(i).data;
		auto current_target_pixel = target_image->row(i).data;
		for (size_t j = 0; j < mask->cols; j++)
		{
			if (*current_mask_pixel != 0)
			{
				*current_target_pixel = LUL[0][*current_target_pixel];
				*(current_target_pixel + 1) = LUL[1][*(current_target_pixel + 1)];
				*(current_target_pixel + 2) = LUL[2][*(current_target_pixel + 2)];
			}

			// Advance to next pixel
			current_target_pixel += 3;
			current_mask_pixel++;
		}
	}
}

JNIEXPORT void JNICALL Java_faceswap_OurNativeLib_pasteFaces(JNIEnv * env, jobject obj, jlong source, jlong face, jlong mask)
{
	cv::Mat *sourceImage = (cv::Mat*)source;
	cv::Mat* faceImage = (cv::Mat*) face;
	cv::Mat* maskImage = (cv::Mat*) mask;
	for (size_t i = 0; i < sourceImage->rows; i++)
	{
		auto frame_pixel = sourceImage->row(i).data;
		auto faces_pixel = faceImage->row(i).data;
		auto masks_pixel = maskImage->row(i).data;

		for (size_t j = 0; j < sourceImage->cols; j++)
		{
			if (*masks_pixel != 0)
			{
				*frame_pixel = ((255 - *masks_pixel) * (*frame_pixel) + (*masks_pixel) * (*faces_pixel)) >> 8; // divide by 256
				*(frame_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(frame_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) >> 8;
				*(frame_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(frame_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) >> 8;
			}

			frame_pixel += 3;
			faces_pixel += 3;
			masks_pixel++;
		}
	}
}

JNIEXPORT jintArray JNICALL Java_faceswap_OurNativeLib_getPointsAndRecDlibColor(JNIEnv * env, jobject obj, jlong matAddr, jint detectorIndex, jint shapeIndex, jint SCALE)
{
	try {
		jintArray rezult = nullptr;
		cv::Mat * inputMat = (cv::Mat*)matAddr;
		frontal_face_detector detector = detectors[detectorIndex];
		shape_predictor sp = predictors[shapeIndex];
		cv::Mat im_small;
		cv::Mat im = *(cv::Mat*)matAddr;
		cv::resize(im, im_small, cv::Size(), 1.0 , 1.0);
		// Make the image larger so we can detect small faces.
		//pyramid_up(img);
		cv_image<bgr_pixel> cimg_small(im_small);
		cv_image<bgr_pixel> cimg(im);

		// Now tell the face detector to give us a list of bounding boxes
		// around all the faces in the image.
		std::vector<rectangle> dets = detector(cimg_small);
		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		rezult = env->NewIntArray(dets.size() * 140);
		jint bodyOfrezult[140];
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			// Resize obtained rectangle for full resolution image. 
			rectangle r(
				(long)(dets[j].left() ),
				(long)(dets[j].top() ),
				(long)(dets[j].right() ),
				(long)(dets[j].bottom())
			);
			full_object_detection shape = sp(cimg, r);
			bodyOfrezult[0] = r.left();
			bodyOfrezult[1] = r.top();
			bodyOfrezult[2] = r.width();
			bodyOfrezult[3] = r.height();
			for (int k = 4, l = 0; k < 140; k += 2, l++)
			{
				dlib::point p = shape.part(l);
				int x = p.x();
				int y = p.y();
				if (x < 0)
				{
					x = 0;
				}
				else
				{
					if (x > im.cols)
					{
						x = im.cols;
					}
				}
				if (y < 0)
				{
					y = 0;
				}else
					if (y > im.rows)
					{
						y = im.rows;
					 }
				bodyOfrezult[k] = x;
				bodyOfrezult[k + 1] = y;
			}
			env->SetIntArrayRegion(rezult, j * 140, 140, bodyOfrezult);
		}
		if (dets.size() == 0)
		{
			rezult = env->NewIntArray(1);
		}
		im.release();
		return rezult;
	}
	catch (exception& e)
	{
		cerr << e.what();
		return 0;
	}
	return 0;
}


