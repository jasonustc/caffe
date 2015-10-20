#include<fstream>
#include<string>
#include<time.h>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "leveldb\db.h"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe\util\math_functions.hpp"
#include "boost\scoped_ptr.hpp"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace caffe;
using boost::scoped_ptr;
using std::string;

const int mnist_image_width = 28;
const int mnist_image_height = 28;
//include sequence length into image channels
//mnist_image_channels = image_channels * sequence_length
const int mnist_image_channels = 20;
const int sample_num = 120000;

void create_db(const string& feat_file, const string& db_name, const int image_width,
	const int image_height, const int image_channels){
	//load feat file
	ifstream in_feat(feat_file);
	CHECK(in_feat.is_open()) << "open " << feat_file << " failed!";

	//create new db
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(options, db_name, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_name <<
		", maybe it already exists.";

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	//save image data into leveldb
	int count = 0;
	clock_t time1, time2;
	time1 = clock();
	Datum image_datum;
	//2 images,each image with 3 channels
	/*
	 *datum: channels, width, height, encoded, data(), float_data()
	 *export: datum.SerializeToString(&out)
	 */
	image_datum.set_channels(image_channels);
	image_datum.set_width(image_width);
	image_datum.set_height(image_height);

	//build training dataset
	int num_images = 0;
	const int label = 1;
	float feat_buf;
	for (int i = 0; i < sample_num; i++){
		image_datum.clear_data();
		image_datum.clear_float_data();
		for (int c = 0; c < image_channels; c++){
			for (int h = 0; h < image_height; h++){
				for (int w = 0; w < image_width; w++){
					in_feat >> feat_buf;
					image_datum.add_float_data(feat_buf);
				}
			}
		//sequential 
		string out;
		image_datum.SerializeToString(&out);
		int length = sprintf_s(key_cstr, kMaxKeyLength, "%09d", i);
		//put into db
		leveldb::Status s = db->Put(leveldb::WriteOptions(), std::string(key_cstr, length), out);
		num_images++;
		if (++count % 100 == 0){
			time2 = clock();
			float diff_time((float)time2 - (float)time1);
			diff_time /= CLOCKS_PER_SEC;
			LOG(INFO) << "Processed " << count << " training images in " << diff_time << " s.";
			LOG(INFO) << "Generated " << num_images << " pairs.";
		}
		}
	}// for (int line_id = 0; line_id < train_end; line_id++)
	LOG(INFO) << "Processed " << num_images << " images";
	delete db;
}

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	if (argc < 3){
		gflags::SetUsageMessage("Convert a set of pair images to leveldb\n"
			"format used as input for caffe.\n"
			"usage: \n"
			" EXE [FLAGS] INPUT_FEAT_FILE DB_NAME\n");
//		LOG(INFO) << gflags::CommandlineFlagsIntoString();
		gflags::ShowUsageWithFlags(argv[0]);
		return 1;
	}

	create_db(argv[1], argv[2], mnist_image_width, mnist_image_height, mnist_image_channels);
	return 0;
}