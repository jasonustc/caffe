Here is the windows version of caffe

Tools:
1. Visual Studio 2013
2. Cuda 6.5
3. OpenCV 2.4.9

Steps:
1. Copy folder $3rdparty and $bin to the caffe root directory
2. Configure the environment variables: $BOOST_1_56_0, $OPENCV_2_4_9
3. Compile the caffe.sln in VS2013

Notes:
1. Currently Caffe works with cuDNN_v1, not cuDNN_v2
2. You need to compile cudnn_*_.cu files firstly manually, then compile the project (I don't know why too...)

You need copy 
More details at https://initialneil.wordpress.com/2015/01/11/build-caffe-in-windows-with-visual-studio-2013-cuda-6-5-opencv-2-4-9
