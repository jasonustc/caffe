#include <string>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace std;
using namespace caffe;
int main()
{
	string mean_file = "W:\\Users\\v-liqing\\caffe\\example\\action_recognition_ucf101\\end2end_frame\\ucf_video_mean.bin";
	BlobProto mean_proto;
	ReadProtoFromBinaryFileOrDie(mean_file, &mean_proto);
	Blob<float> mean_data;
	mean_data.FromProto(mean_proto);

	int n_channels_per_frame = 3;
	int n_frames = mean_proto.channels() / n_channels_per_frame;
	int data_size = n_channels_per_frame*mean_proto.height()*mean_proto.width();

	Blob<float> frame_mean_data;
	frame_mean_data.Reshape(1, n_channels_per_frame, mean_data.height(), mean_data.width());
	float* frame_mean = frame_mean_data.mutable_cpu_data();
	const float* mean = mean_data.cpu_data();
	for (int i = 0; i < frame_mean_data.count(); i++)
	{
		frame_mean[i] = 0;
	}
	for (int i = 1; i < n_frames; i++)
	{
		for (int j = 0; j < data_size; j++)
		{
			frame_mean[j] += mean[i*data_size + j];
		}
	}

	BlobProto modified_data_mean;
	modified_data_mean.CopyFrom(mean_proto);
	for (int i = 0; i < n_frames; i++)
	{
		for (int j = 0; j < data_size; j++)
		{
			modified_data_mean.set_data(i*data_size + j, frame_mean[j]/n_frames);
		}
	}

	string modified_mean_file = "W:\\Users\\v-liqing\\caffe\\example\\action_recognition_ucf101\\end2end_frame\\ucf_video_mean_modify.bin";
	WriteProtoToBinaryFile(modified_data_mean, modified_mean_file);


}