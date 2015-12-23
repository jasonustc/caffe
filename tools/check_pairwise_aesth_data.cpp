#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/db.hpp"
#include "opencv2/opencv.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;

int main(int argc, char** argv){
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);

	if (argc != 3){
		LOG(ERROR) << "Usage: EXE input_leveldb output_db_data_file";
		return 1;
	}

	FILE* output = fopen(argv[2], "w");
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = false;

	LOG(INFO) << "Opening leveldb " << argv[1];
	leveldb::Status status = leveldb::DB::Open(options, argv[1], &db);

	leveldb::ReadOptions read_options;
	read_options.fill_cache = false;
	leveldb::Iterator* it = db->NewIterator(read_options);

	Datum datum;
	int count = 0;

	LOG(INFO) << "Starting Iteration";
	for (it->SeekToFirst(); it->Valid(); it->Next()){
		//just a dummy operation
		datum.ParseFromString(it->value().ToString());
		const std::string& data = datum.data();
		const int float_data_size = datum.float_data_size();
		const int data_size = datum.data().size();
		const int size = std::max<int>(float_data_size, data_size);
		if (count == 0){
			printf("sample size info:\n");
			LOG(INFO) << "channels: " << datum.channels();
			LOG(INFO) << "width: " << datum.width();
			LOG(INFO) << "height: " << datum.height();
		}
		//two pair images
		CHECK_EQ(size, datum.channels() * datum.width() * datum.height());
		if (count == 0){
			fprintf(output, "[C, W, H]: %d %d %d\n", datum.channels(), datum.width(), datum.height());
		}

		for (int i = 0; i < float_data_size; i++){
			fprintf(output, "%f ", static_cast<float>(datum.float_data(i)));
		}
		for (int i = 0; i < data_size; i++){
			//image pixel data is read as char 
			//and one element are corresponding to one pixel
			//TODO: make the visualize of data much better here.
			fprintf(output, "%d ", static_cast<unsigned int>(datum.data()[i]));
		}
		
		++count;
		if (count % 1000 == 0){
			LOG(ERROR) << "Have read: " << count << " files.";
		}
		fprintf(output, "\n");
	}
	if (count % 1000 != 0){
		LOG(INFO) << "Processed " << count << " files.";
	}
	fclose(output);
	delete db;
	return 0;
}
