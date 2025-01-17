# Motion analysis using optical flow

## Description

This project aims to aproximate and show the general direction of moving objects in a given sequence of bitmap images or a video. The result of this aproximation gives the moving objects a color code representing the direction in which they are heading.
<br>
The poject uses OpenCV's Optical Flow Dense to estimate the direction of the moving objects and Region Growing applied on this set of directions to better predict the angle in which the objects move. <br>

## Usage

The program starts by presenting the user 2 options: 
1. Use the algorithm on a video file ( .avi )
2. Use the algorithm on a bitmap file sequance ( .bmp files )
After that the first frame will appear on the screen. <br>
In order to see the result of the algorithm the user needs to press any key (except ESC) to proceed to the next frame. <br>
Finally, the user will se both the raw optical flow dense aproximation and the proccesed optical flow dense along with a line that indicates the direction of the moving object. 

## Requirements

The source code provided needs the following, in order to run:
- Microsoft Visual Studio
- OpenCV library
- Either a bitmat sequence stored in a folder or a video.

## Contents

The repository contains the source code and the additional functions that from the algorithm. <br>
The documentation of this project.

## Instalation

1. Clone the repository
2. Create an empty VS project that has the OpenCV libray linked.
3. Copy the source from ```MotionAnaliysisUsingOpticalFlow\Source Code``` into the empy project
4. Run the program.


