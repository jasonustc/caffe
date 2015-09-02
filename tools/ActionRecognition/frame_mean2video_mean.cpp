#include <string>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace std;
using namespace caffe;
int main(int argc, char** argv)
{
	const int num_required_args = 3;
	if (argc < num_required_args)
	{
		cout << "Usage:\nEXE frame_mean frames_per_video video_mean";
		return -1;
	}
	string path_frame_mean = argv[1];
	BlobProto proto_frame_mean;
	ReadProtoFromBinaryFileOrDie(path_frame_mean, &proto_frame_mean);
	Blob<float> blob_frame_mean;
	blob_frame_mean.FromProto(proto_frame_mean);
	const float* data_frame_mean = blob_frame_mean.cpu_data();
	const int n_channels_per_frame = blob_frame_mean.channels();
	int n_frames = atoi(argv[2]);

	int data_size = n_channels_per_frame*blob_frame_mean.height()*blob_frame_mean.width();

	BlobProto proto_video_mean;
	proto_video_mean.set_num(1);
	proto_video_mean.set_channels(n_channels_per_frame*n_frames);
	proto_video_mean.set_height(blob_frame_mean.height());
	proto_video_mean.set_width(blob_frame_mean.width());
	for (int i = 0; i < n_frames; i++)
	{
		for (int j = 0; j < data_size; j++)
		{
			proto_video_mean.add_data(data_frame_mean[j]);
		}
	}

	string path_video_mean = argv[3];
	WriteProtoToBinaryFile(proto_video_mean, path_video_mean);
	WriteProtoToTextFile(proto_frame_mean, "frame_mean.txt");
	WriteProtoToTextFile(proto_video_mean, "video_mean.txt");


}