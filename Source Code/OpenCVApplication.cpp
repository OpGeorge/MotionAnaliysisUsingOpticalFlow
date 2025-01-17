// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "Functions.h"

wchar_t* projectPath;



int Nx[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
int Ny[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
bool compareByY(Point a, Point b) {
	return a.y < b.y;
}

Mat ReginGrowRetuningOutput(Mat input, int xStart, int yStart, Mat* processMask) {

	Mat output = input.clone();

	Mat temp = input.clone();
	Mat labels = Mat::zeros(temp.size(), CV_16UC1);
	Mat dst = Mat::zeros(temp.size(), CV_8UC1);
	queue<Point> que;
	int k = 1;
	int N = 1;
	que.push(Point(yStart, xStart));
	//float Hue_avg = temp.at<uchar>(yStart, xStart);
	int dirDeg_avg = temp.at<ushort>(yStart, xStart);
	//Vec3f color_avg = inputOutput.at<Vec3b>(yStart, xStart);
	int T = 15;

	while (!que.empty())
	{
		Point oldest = que.front();
		que.pop();
		int yy = oldest.x;
		int xx = oldest.y;

		for (int i = 0; i < 8; i++) {
			int ny = yy + Ny[i];
			int nx = xx + Nx[i];
			if (ny >= 0 && ny < temp.rows &&
				nx >= 0 && nx < temp.cols && 
				temp.at<ushort>(ny, nx) >= 345 && temp.at<ushort>(ny, nx) <= 360)
			{
				
				temp.at<ushort>(ny, nx) = 360 - temp.at<ushort>(ny, nx);
			}

			if (ny >= 0 && ny < temp.rows && nx >= 0 && nx < temp.cols &&
				abs(temp.at<ushort>(ny, nx) - dirDeg_avg) < T &&
				labels.at<uint16_t>(ny, nx) == 0)
			{
				que.push(Point(ny, nx));
				labels.at<uint16_t>(ny, nx) = k;

				(*processMask).at<uchar>(ny, nx) = 1; // Mark as processed for the outside processing

				dirDeg_avg = (N * dirDeg_avg + temp.at<ushort>(ny, nx)) / (N + 1);
				N++;
			}
		}
	}


	for (int i = 0; i < labels.rows; i++)
		for (int j = 0; j < labels.cols; j++)
		{
			if (labels.at<uint16_t>(i, j) == k) {
				dst.at<uchar>(i, j) = 255;
				output.at<ushort>(i, j) = dirDeg_avg;
			}


		}


	Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3));
	erode(dst, dst, element1, Point(-1, -1), 4);

	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(dst, dst, element2, Point(-1, -1), 4);

	

	//imshow("dst", dst);
	//imshow("UNIFORM", output);
	return output;


}

Mat ReginGrowRetuning(Mat inputOutput, int xStart, int yStart, Mat param, Mat* processMask) {


	Mat temp = param.clone();
	Mat labels = Mat::zeros(temp.size(), CV_16UC1);
	Mat dst = Mat::zeros(temp.size(), CV_8UC1);
	queue<Point> que;
	int k = 1;
	int N = 1;
	que.push(Point(yStart, xStart));
	float Hue_avg = temp.at<uchar>(yStart, xStart);
	Vec3f color_avg = inputOutput.at<Vec3b>(yStart, xStart);
	int T = 12;

	while (!que.empty())
	{
		Point oldest = que.front();
		que.pop();
		int yy = oldest.x;
		int xx = oldest.y;

		for (int i = 0; i < 8; i++) {
			int ny = yy + Ny[i];
			int nx = xx + Nx[i];
			if (ny >= 0 && ny < temp.rows && nx >= 0 && nx < temp.cols &&
				abs(temp.at<uchar>(ny, nx) - Hue_avg) < T &&
				labels.at<uint16_t>(ny, nx) == 0 && (*processMask).at<uchar>(ny,nx) == 0 )
			{
				que.push(Point(ny, nx));
				labels.at<uint16_t>(ny, nx) = k;
				
				(*processMask).at<uchar>(ny, nx) = 1; // Mark as processed

				Hue_avg = (N * Hue_avg + temp.at<uchar>(ny, nx)) / (N + 1);

				Vec3b currentColor = inputOutput.at<Vec3b>(ny, nx);
				for (int c = 0; c < 3; c++) {
					color_avg[c] = (N * color_avg[c] + currentColor[c]) / (N + 1);
				}

				N++;
			}
		}
	}


	for (int i = 0; i < labels.rows; i++)
		for (int j = 0; j < labels.cols; j++)
		{
			if (labels.at<uint16_t>(i, j) == k) {
				dst.at<uchar>(i, j) = 255;
				inputOutput.at<Vec3b>(i, j) = Vec3b(color_avg);
			}


		}


	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(dst, dst, element2, Point(-1, -1), 4);

	Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3));
	erode(dst, dst, element1, Point(-1, -1), 4);

	imshow("dst", dst);
	imshow("COLORED", inputOutput);
	return inputOutput;


}



void MyCallBackFuncRG(int event, int xStart, int yStart, int flags, void* param) // Region Growing alg and calculations
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat temp = (*src).clone();
		Mat labels = Mat::zeros(temp.size(), CV_16UC1);
		Mat dst = Mat::zeros(temp.size(), CV_8UC1);
		queue<Point> que;
		int k = 1;
		int N = 1;
		que.push(Point(yStart, xStart));
		float Hue_avg = temp.at<uchar>(yStart, xStart);
		int T = 12;

		while (!que.empty())
		{
			Point oldest = que.front();
			que.pop();
			int yy = oldest.x;
			int xx = oldest.y;

			for (int i = 0; i < 8; i++) {
				int ny = yy + Ny[i];
				int nx = xx + Nx[i];
				if (ny >= 0 && ny < temp.rows && nx >= 0 && nx < temp.cols &&    // verifica daca nu iasa din poza vecinul
					abs(temp.at<uchar>(ny, nx) - Hue_avg) < T && //verifica daca pixelul este cam de aceeasi culoare ca si avg
					labels.at<uint16_t>(ny, nx) == 0) // verifica daca nu s-a mai trecut pe acolo   0 - nu a mai trecut  1 - a mai trecut
				{
					que.push(Point(ny, nx)); // pixelul pe care acum il marchez ca ii bun tre sa il pun in coada si dupa sa ii vad vecini daca is buni
					labels.at<uint16_t>(ny, nx) = k;
					Hue_avg = (N * Hue_avg + temp.at<uchar>(ny, nx)) / (N + 1); // adun peste val initiala (am a1 la inceput cu N1 dupa am a1 + a2 si N1 + 1 = N2 cu avg (a1+a2)/N2 smad ca sa am un Hue_avg cat mai precis )
					N++;
				}
			}
		}


		for (int i = 0; i < labels.rows; i++)
			for (int j = 0; j < labels.cols; j++)
			{
				if (labels.at<uint16_t>(i, j) == k)
					dst.at<uchar>(i, j) = 255; // aici daca ii sa-l fac pe culori trebuie sa pun ca arg ce culare sa ii dau
					// si ceea ce vreau sa zic ii ca trebuie sa translatez coord din labes in coord din imaginea intiala
					// aici dst e all zeros

			}


		Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(dst, dst, element2, Point(-1, -1), 4);

		Mat element1 = getStructuringElement(MORPH_CROSS, Size(3, 3)); 
		erode(dst, dst, element1, Point(-1, -1), 4);
		// prin op morfologice de mai sus rafinez rezultatul final


		imshow("dst", dst);
		waitKey();
	}
}

void MyRegionGrowing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);
		GaussianBlur(rgb, rgb, Size(5, 5), 0, 0);
		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);
		split(hsv, channels);

		Mat H = channels[0] * 255 / 180;
		imshow("My Window", rgb);
		//ininte de toate trebuie sa procesez imaginea cu functia de lab7
		setMouseCallback("My Window", MyCallBackFuncRG, &H);
		//fac region growing tot pe canalul H
		waitKey();
	}
}

void MyRegionGrowingV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat rgb = imread(fname);
		GaussianBlur(rgb, rgb, Size(5, 5), 0, 0);
		Mat hsv;
		Mat channels[3];
		cvtColor(rgb, hsv, COLOR_BGR2HSV);
		split(hsv, channels);

		Mat H = channels[0] * 255 / 180;
		Mat V = channels[2];
		imshow("My Window", rgb);
		setMouseCallback("My Window", MyCallBackFuncRG, &V);
		waitKey();
	}
}

vector<Point> getLocalMax(int hist_dir[360], int rows, int cols) {

	vector<Point> localMax;
	int yTh = 0;
	for (int i = 0; i < 360; i++) {
		if (hist_dir[i] > yTh) {
			yTh = hist_dir[i];
		}
	}
	printf("\n\nAici e yTH: %d\n\n", yTh);

	bool found = false;
	int nrOfPeaks = 20;
	while (!found) {
		int i = 0;
		int nr = 0;
		localMax.erase(localMax.begin(), localMax.end());
		while (i < 360 && nr < nrOfPeaks && !found) {
			if (hist_dir[i] > yTh) {
				nr++;
				localMax.push_back(Point(i, hist_dir[i]));
			}
			if (nr == nrOfPeaks) {
				found = true;
				printf("\n nr reached : %d when found = rue\n", nr);
			}
			i++;

		}
		yTh--;

	}
	printf("\n\n y TH = %d\n\n", yTh);

	int distTh = 3;
	
	if (localMax.size() == 0) {
		throw runtime_error("nu ai val in localMax");
	}
	vector<Point> finalLocalMax;
	
	printf("\n\nNR DE MAXIME:%d\n\n", localMax.size());
	for (int i = 0; i < localMax.size(); i++) {
		printf(" local[ %d ]=> x:%d y:%d ", i, localMax[i].x,localMax[i].y);
	}

	for (int i = 1; i < localMax.size() - 1; i++) {
		if (localMax[i].x - localMax[i - 1].x > distTh 
				|| localMax[i+1].x - localMax[i].x > distTh) {
			finalLocalMax.push_back(localMax[i]);
		}
	}
	return finalLocalMax;

}

void opticalFlowFarneback() {

	makeColorwheel(); // initaializes the colorwhel for the colorcode module
	make_HSI2RGB_LUT();
	Mat crnt; // current frame red as grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat flow; // flow - matrix containing the optical flow vectors/pixel

	Mat flowRegionGrow;

	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	char c;
	double minVel = 0.7;
	
	Mat frame;

	VideoCapture cap("Videos/laboratory.AVI");

	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	
	
	int frameNum = -1; //current frame counter
	while (cap.read(frame))// citeste in fname numele caii complete
		// la cate un fisier bitmap din secventa
	{
		
		++frameNum;
		if (frameNum == 0) {
			imshow("sursa", frame);
		}
		
		cvtColor(frame, crnt, COLOR_BGR2GRAY);
		
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		
		if (frameNum > 0) // not the first frame
		{

			int winSize = 15;
			double t = (double)getTickCount();

			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, winSize, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

			flowRegionGrow = ReturnFlowDense("DST", crnt, flow, minVel, true);
			// Hue calculation
			//GaussianBlur(flowRegionGrow, flowRegionGrow, Size(5, 5), 0, 0);
			Mat hsv;
			Mat channels[3];
			cvtColor(flowRegionGrow, hsv, COLOR_BGR2HSV);
			split(hsv, channels);

			Mat H = channels[0] * 255 / 180;

			//imshow("HUE MAT OF FLOW", H);
			//imshow("Returned flow Dense", flowRegionGrow);


			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);

			int hist_dir[360] = { 0 };
			for (int i = 0; i < 360; i++) {
				hist_dir[i] = 0;
			}

			Mat processedMask = Mat::zeros(flow.size(), CV_8U);

			Mat dirDegMat = Mat::zeros(flow.size(),CV_16UC1);

			Mat binaryMat = Mat::zeros(flow.size(), CV_8U);


			for (int r = 0; r < flow.rows; r++) {
				for (int c = 0; c < flow.cols; c++) {

					Point2f f = flow.at<Point2f>(r, c);
					float dir_rad = PI + atan2(-f.y, -f.x);
					int dir_deg = dir_rad * 180 / PI;
					float magnitude = sqrt(f.x * f.x + f.y * f.y);
					
					if (magnitude > minVel ) {
						dirDegMat.at<ushort>(r, c) = dir_deg;
						binaryMat.at<uchar>(r, c) = 1;
					}
					else {
						dirDegMat.at<ushort>(r, c) = 500;
						binaryMat.at<uchar>(r, c) = 0;
					}

				}

			}

			//imshow("NEUNIFORM", dirDegMat);

			Mat output;
			for (int r = 0; r < flow.rows; r++) {
				for (int c = 0; c < flow.cols; c++) {

					Point2f f = flow.at<Point2f>(r, c);
					float dir_rad = PI + atan2(-f.y, -f.x);
					int dir_deg = dir_rad * 180 / PI;
					float magnitude = sqrt(f.x * f.x + f.y * f.y);
					// daca magintude ii mai mare decat min vel atunci fa regin grow pe pixelul 
					// dau culoarea medie a regiunii
					if (magnitude > minVel && processedMask.at<uchar>(r,c) == 0){
						// TO DO : salveza pozitia pixelului care indeplineste conditia
						//done
						//flowRegionGrow = ReginGrowRetuning(flowRegionGrow, c, r, H);
						processedMask.at<uchar>(r, c) = 1;
						
						//flowRegionGrow = ReginGrowRetuning(flowRegionGrow, c, r, H, &processedMask);
						output = ReginGrowRetuningOutput(dirDegMat, c, r, &processedMask);

						hist_dir[dir_deg]++;
					}

				}

			}

			//showHistogram("Hist", hist_dir, 360, 200, true);
			//showHistogramDir("Hist_Dir", hist_dir, 360, 200, true);
			
			Mat_<Vec3b> coloredDirections = frame.clone();

			vector<vector<Point>> contours;

			applyMorpOp(binaryMat, &output,&contours);
			

			for (int i = 0; i < coloredDirections.rows; i++) {
				for (int j = 0; j < coloredDirections.cols; j++) {
					if (output.at<ushort>(i, j) < 500) {
						coloredDirections(i, j) = (Vec3b)getColorFromDir(output.at<ushort>(i, j));
					}
				}
			}

			for (const auto& contour : contours) {
				Moments m = moments(contour, false);
				double cX = m.m10 / m.m00;
				double cY = m.m01 / m.m00;
				int lenght = 100;
				int thickness = 2;
				if (output.at<ushort>(cY, cX) == 0 || output.at<ushort>(cY, cX) == 360) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) > 0 && output.at<ushort>(cY, cX) < 90) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY + lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) == 90)
				{
					line(coloredDirections, Point(cX, cY), Point(cX, cY + lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) > 90 && output.at<ushort>(cY, cX) < 180) {
				
					line(coloredDirections, Point(cX, cY), Point(cX - lenght, cY + lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) == 180) {
					line(coloredDirections, Point(cX, cY), Point(cX - lenght, cY), coloredDirections(cY, cX), thickness, 8, 0);
				}

				if (output.at<ushort>(cY, cX) > 180 && output.at<ushort>(cY, cX) < 270)
				{
					line(coloredDirections, Point(cX, cY), Point(cX - lenght, cY - lenght), coloredDirections(cY, cX), thickness, 8, 0);

				}
				if (output.at<ushort>(cY, cX) == 270) {
					line(coloredDirections, Point(cX, cY), Point(cX , cY - lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) > 270 && output.at<ushort>(cY, cX) < 345) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY - lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}

				if (output.at<ushort>(cY, cX) >= 345 && output.at<ushort>(cY, cX) < 360) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY), coloredDirections(cY, cX), thickness, 8, 0);
				}
				

				// (cX, cY) is the centroid of the contour
			}
			

			imshow("COLORED DIRS", coloredDirections);

			

			showFlowDense("DST", crnt, flow, minVel, true);
			//imshow("Flow Region Grow", flowRegionGrow);
			



		}
		// store crntent frame as previos for the next cycle
		imshow("OG", crnt);
		prev = crnt.clone();
		c = waitKey(0); // press any key to advance between frames //for continous play use cvWaitKey( delay > 0) 
		if (c == 27) { // press ESC to exit 
			printf("ESC pressed - playback finished\n\n");
			break; //ESC pressed 
		}
	}

}

void opticalFlowFarnebackBitmapSequence() {



	makeColorwheel(); // initaializes the colorwhel for the colorcode module
	make_HSI2RGB_LUT();
	Mat crnt; // current frame red as grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat flow; // flow - matrix containing the optical flow vectors/pixel
	char folderName[MAX_PATH];
	char fname[MAX_PATH];
	char c;
	int minVel = 1;

	Mat frame;
	Mat flowRegionGrow;

	if (openFolderDlg(folderName) == 0)
		return;
	FileGetter fg(folderName, "bmp");
	int frameNum = -1; //current frame counter
	while (fg.getNextAbsFile(fname))// citeste in fname numele caii complete
		// la cate un fisier bitmap din secventa
	{
		crnt = imread(fname, IMREAD_GRAYSCALE);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);

	
		++frameNum;
		if (frameNum == 0) {
			imshow("sursa", crnt);
		}

		//cvtColor(frame, crnt, COLOR_BGR2GRAY);

		//GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);

		if (frameNum > 0) // not the first frame
		{

			int winSize = 15;
			double t = (double)getTickCount();

			calcOpticalFlowFarneback(prev, crnt, flow, 0.5, 3, winSize, 10, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);

			flowRegionGrow = ReturnFlowDense("DST", crnt, flow, minVel, true);
			// Hue calculation
			//GaussianBlur(flowRegionGrow, flowRegionGrow, Size(5, 5), 0, 0);
			Mat hsv;
			Mat channels[3];
			cvtColor(flowRegionGrow, hsv, COLOR_BGR2HSV);
			split(hsv, channels);

			Mat H = channels[0] * 255 / 180;

			//imshow("HUE MAT OF FLOW", H);
			//imshow("Returned flow Dense", flowRegionGrow);


			t = ((double)getTickCount() - t) / getTickFrequency();
			printf("%d - %.3f [ms]\n", frameNum, t * 1000);

			int hist_dir[360] = { 0 };
			for (int i = 0; i < 360; i++) {
				hist_dir[i] = 0;
			}

			Mat processedMask = Mat::zeros(flow.size(), CV_8U);

			Mat dirDegMat = Mat::zeros(flow.size(), CV_16UC1);

			Mat binaryMat = Mat::zeros(flow.size(), CV_8U);


			for (int r = 0; r < flow.rows; r++) {
				for (int c = 0; c < flow.cols; c++) {

					Point2f f = flow.at<Point2f>(r, c);
					float dir_rad = PI + atan2(-f.y, -f.x);
					int dir_deg = dir_rad * 180 / PI;
					float magnitude = sqrt(f.x * f.x + f.y * f.y);

					if (magnitude > minVel) {
						dirDegMat.at<ushort>(r, c) = dir_deg;
						binaryMat.at<uchar>(r, c) = 1;
					}
					else {
						dirDegMat.at<ushort>(r, c) = 500;
						binaryMat.at<uchar>(r, c) = 0;
					}

				}

			}

			//imshow("NEUNIFORM", dirDegMat);

			Mat output;
			for (int r = 0; r < flow.rows; r++) {
				for (int c = 0; c < flow.cols; c++) {

					Point2f f = flow.at<Point2f>(r, c);
					float dir_rad = PI + atan2(-f.y, -f.x);
					int dir_deg = dir_rad * 180 / PI;
					float magnitude = sqrt(f.x * f.x + f.y * f.y);
					// daca magintude ii mai mare decat min vel atunci fa regin grow pe pixelul 
					// dau culoarea medie a regiunii
					if (magnitude > minVel && processedMask.at<uchar>(r, c) == 0) {
						// TO DO : salveza pozitia pixelului care indeplineste conditia
						//done
						//flowRegionGrow = ReginGrowRetuning(flowRegionGrow, c, r, H);
						processedMask.at<uchar>(r, c) = 1;

						//flowRegionGrow = ReginGrowRetuning(flowRegionGrow, c, r, H, &processedMask);
						output = ReginGrowRetuningOutput(dirDegMat, c, r, &processedMask);

						hist_dir[dir_deg]++;
					}

				}

			}
			//frame
			//showHistogram("Hist", hist_dir, 360, 200, true);
			//showHistogramDir("Hist_Dir", hist_dir, 360, 200, true);

			Mat_<Vec3b> coloredDirections;
				
			cvtColor(crnt, coloredDirections,COLOR_GRAY2BGR);

			vector<vector<Point>> contours;

			applyMorpOp(binaryMat, &output, &contours);


			for (int i = 0; i < coloredDirections.rows; i++) {
				for (int j = 0; j < coloredDirections.cols; j++) {
					if (output.at<ushort>(i, j) < 500) {
						coloredDirections(i, j) = (Vec3b)getColorFromDir(output.at<ushort>(i, j));
					}
				}
			}

			for (const auto& contour : contours) {
				Moments m = moments(contour, false);
				double cX = m.m10 / m.m00;
				double cY = m.m01 / m.m00;
				int lenght = 100;
				int thickness = 2;
				if (output.at<ushort>(cY, cX) == 0 || output.at<ushort>(cY, cX) == 360) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) > 0 && output.at<ushort>(cY, cX) < 90) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY + lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) == 90)
				{
					line(coloredDirections, Point(cX, cY), Point(cX, cY + lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) > 90 && output.at<ushort>(cY, cX) < 180) {

					line(coloredDirections, Point(cX, cY), Point(cX - lenght, cY + lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) == 180) {
					line(coloredDirections, Point(cX, cY), Point(cX - lenght, cY), coloredDirections(cY, cX), thickness, 8, 0);
				}

				if (output.at<ushort>(cY, cX) > 180 && output.at<ushort>(cY, cX) < 270)
				{
					line(coloredDirections, Point(cX, cY), Point(cX - lenght, cY - lenght), coloredDirections(cY, cX), thickness, 8, 0);

				}
				if (output.at<ushort>(cY, cX) == 270) {
					line(coloredDirections, Point(cX, cY), Point(cX, cY - lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}
				if (output.at<ushort>(cY, cX) > 270 && output.at<ushort>(cY, cX) < 345) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY - lenght), coloredDirections(cY, cX), thickness, 8, 0);
				}

				if (output.at<ushort>(cY, cX) >= 345 && output.at<ushort>(cY, cX) < 360) {
					line(coloredDirections, Point(cX, cY), Point(cX + lenght, cY), coloredDirections(cY, cX), thickness, 8, 0);
				}


				// (cX, cY) is the centroid of the contour
			}


			imshow("COLORED DIRS", coloredDirections);



			showFlowDense("DST", crnt, flow, minVel, true);
			//imshow("Flow Region Grow", flowRegionGrow);




		}
		// store crntent frame as previos for the next cycle
		imshow("OG", crnt);
		prev = crnt.clone();
		c = waitKey(0); // press any key to advance between frames //for continous play use cvWaitKey( delay > 0) 
		if (c == 27) { // press ESC to exit 
			printf("ESC pressed - playback finished\n\n");
			break; //ESC pressed 
		}
	}

}



int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Optical Flow Farneback & Applied Region Growing\n");
		printf(" 2 - Optical Flow Farneback & Applied Region Growing Bitmap sequence\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			
			
			case 1:
				opticalFlowFarneback();
			case 2:
				opticalFlowFarnebackBitmapSequence();
			default:
				break;
		}
	}
	while (op!=0);
	return 0;
}