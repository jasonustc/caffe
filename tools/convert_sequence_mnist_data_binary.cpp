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

DEFINE_int32(image_channels, 28, "The channels * sequence length value");
DEFINE_int32(image_height, 28, "The image height of each frame");
DEFINE_int32(image_width, 28, "The image width of each frame");
DEFINE_int32(num_samples, 60000, "Num of samples in the dataset");

void create_db(const string& feat_file, const string& label_file, const string& db_name, 
	const int image_width, const int image_height, 
	const int image_channels, const int num_samples){
	//load feat file
	ifstream in_feat(feat_file);
	CHECK(in_feat.is_open()) << "open " << feat_file << " failed!";
	ifstream in_label(label_file);
	CHECK(in_label.is_open()) << "open " << label_file << "failed!";

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
	image_datum.set_channels(image_height);
	image_datum.set_width(1);
	image_datum.set_height(image_width);

	//build training dataset
	int label;
	float feat_buf;
	//put one sequence into one datum, with sequence length in channels
	for (int i = 0; i < num_samples; i++){
		image_datum.clear_data();
		image_datum.clear_float_data();
		in_label >> label;
		image_datum.set_label(label);
		//28 row => 28 frames in sequence
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
		if (++count % 100 == 0){
			time2 = clock();
			float diff_time((float)time2 - (float)time1);
			diff_time /= CLOCKS_PER_SEC;
			LOG(INFO) << "Processed " << count << " training images in " << diff_time << " s.";
		}
	}// for (int line_id = 0; line_id < train_end; line_id++)
	LOG(INFO) << "Processed " << count << " images";
	in_feat.close();
	in_label.close();
	delete db;
}

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::LogToStderr();
	if (argc < 4){
		gflags::SetUsageMessage("Convert a set of pair images to leveldb\n"
			"format used as input for caffe.\n"
			"usage: \n"
			" EXE [FLAGS] INPUT_FEAT_FILE INPUT_LABEL_FILE DB_NAME\n");
//		LOG(INFO) << gflags::CommandlineFlagsIntoString();
		gflags::ShowUsageWithFlags(argv[0]);
		return 1;
	}

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	create_db(argv[1], argv[2], argv[3], FLAGS_image_width, FLAGS_image_height, 
		FLAGS_image_channels, FLAGS_num_samples);
	return 0;
}