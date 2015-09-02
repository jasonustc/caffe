// Copyright 2014 BVLC and contributors.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;

int main(int argc, char** argv) 
{
	::google::InitGoogleLogging(argv[0]);

	if (argc != 3) {
		LOG(ERROR) << "Usage: leveldb_read input_leveldb output_file";
		return 1;
	}
	FILE* output=fopen(argv[2],"w");
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = false;

	LOG(INFO) << "Opening leveldb " << argv[1];
	leveldb::Status status = leveldb::DB::Open(options, argv[1], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[1];

	leveldb::ReadOptions read_options;
	read_options.fill_cache = false;
	leveldb::Iterator* it = db->NewIterator(read_options);

	Datum datum;
	int count = 0;

	LOG(INFO) << "Starting Iteration";
	for (it->SeekToFirst(); it->Valid(); it->Next()) 
	{
		// just a dummy operation
		datum.ParseFromString(it->value().ToString());
		const string& data = datum.data();
		int size_in_datum = std::max<int>(datum.data().size(), datum.float_data_size());
		printf("size:%d\n",size_in_datum);
		for (int i = 0; i < size_in_datum; ++i)
		{
			fprintf(output, "%f ", static_cast<float>(datum.float_data(i)));
		}

		++count;
		if (count % 1000 == 0)
		{
			LOG(ERROR) << "Have read: " << count << " files.";
		}
		fprintf(output, "\n");
	}
	if (count % 1000 != 0) 
	{
		LOG(ERROR) << "Processed " << count << " files.";
	}
	fclose(output);
	delete db;
	return 0;
}
