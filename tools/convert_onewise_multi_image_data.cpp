//added by qing li 2014-12-27
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

using namespace std;
using namespace caffe;


bool ReadImageListToDatum(const vector<string>& list_file, const string& img_dir,
	const int label, const int height, const int width, const bool is_color,
	const std::string& encoding, Datum* datum)
{
	int n_img = list_file.size();
	cv::Mat  merge_img= ReadImageToCVMat(img_dir + '\\' + list_file[0] + ".jpg", height, width, is_color);
	int one_channels = merge_img.channels();
	int rows = merge_img.rows;
	int cols = merge_img.cols;
	int merge_channels=one_channels*n_img;

	merge_img.create(rows, cols, CV_8UC(merge_channels));
	cv::Mat one_img;
	for (int img_id = 0; img_id < list_file.size(); img_id++)
	{
		one_img=ReadImageToCVMat(img_dir + '\\' + list_file[img_id] + ".jpg", height, width, is_color);
		
		for (int ri = 0; ri < rows; ri++)
		{
			auto one_ptr = one_img.ptr<uchar>(ri);
			auto merge_ptr = merge_img.ptr<uchar>(ri);
			for (int ci = 0; ci < cols; ci++)
			{
				for (int ch = 0; ch <one_channels; ch++)
				{
					merge_ptr[ci*merge_channels + img_id*one_channels + ch] = one_ptr[ci*one_channels + ch];
				}

			}

		}
	}

	if (merge_img.data)
	{
		if (encoding.size())
		{
			std::vector<uchar> buf;
			cv::imencode("." + encoding, merge_img, buf);
			datum->set_data(string(reinterpret_cast<char*>(&buf[0]), buf.size()));
			datum->set_label(label);
			datum->set_encoded(true);
			return true;
		}

		CVMatToDatum(merge_img, datum);
		datum->set_label(label);
		return true;
	}
	else
	{
		return false;

	}

}

void convert_dataset(const char* video2img_list_filename, const char* video_list_filename,
        const char* db_filename, int resize_width, int resize_height, int shuffle, string img_dir) {
  // Open files
  std::ifstream video2img_list_in(video2img_list_filename, std::ios::in);
  std::ifstream video_list_in(video_list_filename, std::ios::in);
  CHECK(video2img_list_in) << "Unable to open file " << video2img_list_filename;
  CHECK(video_list_in) << "Unable to open file " << video_list_filename;
  
  //read metadata
  map<string, vector<string>> video2img_list;
  string video;
  string img;
  while(video2img_list_in>>video>>img)
  {
	  if (video2img_list.find(video) == video2img_list.end())
		  video2img_list.insert(pair<string, vector<string>>(video, vector<string>()));
	  video2img_list[video].push_back(img);
  }

  //check whether all videos have the same amounts of frames
  map<string, vector<string>>::iterator it = video2img_list.begin();
  int frames_per_video = it->second.size();
  it++;
  for (; it != video2img_list.end(); it++)
  {
	  CHECK(it->second.size() == frames_per_video)
		  << it->first << " has different amounts of frames.";
  }


  int label;
  std::vector<pair<string, int>> video_list;
  while(video_list_in>>video>>label)
  {
    video_list.push_back(make_pair(video, label));
  }

  video2img_list_in.close();
  video_list_in.close();
 

  if(shuffle==1)
  {
	  LOG(INFO)<<"Shuffling data";
	  std::random_shuffle(video_list.begin(), video_list.end());
  }
   int num_videos=video_list.size();

  LOG(INFO)<<"A total of "<<num_videos<<" videos";

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  options.write_buffer_size = 256 * 1024 * 1024;
  options.max_open_files = 2000;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";


  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  const int kMaxKeyLength = 256;
  char key[kMaxKeyLength];
  std::string value;

  std::string enc = "";
  int count = 0;
  for (int videoid = 0; videoid < num_videos; ++videoid) {
	  Datum merge_datum;
	  video = video_list[videoid].first;
	  label = video_list[videoid].second;
	  vector<string> img_list1 = video2img_list[video];
	  ReadImageListToDatum(img_list1, img_dir, label, resize_height, resize_width, true, enc, &merge_datum);
	  //merge_datum.set_label(label);
	  //merge_datum.set_channels(img_list1.size() * 3);
	  //merge_datum.set_width(resize_width);
	  //merge_datum.set_height(resize_height);
	  //merge_datum.set_encoded(true);
	  ////char* buffer = new char[merge_datum.channels()*merge_datum.width()*merge_datum.height()];
	  //string buffer;
	  ////std::string buffer(size, ' ');
	  //for (int i = 0; i < img_list1.size(); i++)
	  //{
		 // string img = img_list1[i];
		 // Datum one_img_datum;
		 // bool status;
		 // status = ReadImageToDatum(img_dir+'\\'+img+".jpg", 1, resize_height, resize_width, true,
			//  enc, &one_img_datum);
		 // CHECK(status == true) << "fail to read image:" << img << '\n';
		 // const string& data = one_img_datum.data();
		 // buffer.append(data);
	  //}
	  //merge_datum.set_data(buffer);
	  merge_datum.SerializeToString(&value);
	  int length = sprintf_s(key, kMaxKeyLength, "%08d", videoid);
	  batch->Put(string(key, length),value);
		if ((++count)% 100 == 0)
		{
			LOG(INFO) << "proccessd:" <<count;
			db->Write(leveldb::WriteOptions(), batch);
			delete batch;
			batch = new leveldb::WriteBatch();
		}
  }
    
	//write the last batch
	if (count % 100 != 0)
	{
		LOG(INFO) << "proccessd:" <<count;
		db->Write(leveldb::WriteOptions(), batch);
		delete batch;
	}

  delete db;
}


int main(int argc, char** argv)
{
	if (argc!=8)
	{
    printf("Usage:\n"
           "    EXE video2img_list video_video_list output_db_file img_width img_height RANDOM_SHUFFLE_DATA[0 or 1] img_dir"
           "\n"
           );
	}
	else {
		google::InitGoogleLogging(argv[0]);
		int width = atoi(argv[4]);
		int height = atoi(argv[5]);
		int shuffle = atoi(argv[6]);
		string img_dir(argv[7]);
		convert_dataset(argv[1], argv[2], argv[3], width, height, shuffle, img_dir);
	}
    return 0;
}
