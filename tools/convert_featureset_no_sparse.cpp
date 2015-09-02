//added by Qing Li, 2014-12-13
// This program converts a set of features to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_featureset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the features, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.fisher 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 5) {
    printf("Convert a set of features to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
		"    convert_featureset feature_set_file(no_sparse,binary) label_file(txt)  DB_NAME frame_per_video"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
    return 1;
  }
  std::ifstream if_feature_set(argv[1], std::ios::binary);
  std::ifstream if_label_set(argv[2]);
  
  int row, col, data_type;
  if_feature_set.read((char*)&row, sizeof(int));
  if_feature_set.read((char*)&col, sizeof(int));
  if_feature_set.read((char*)&data_type, sizeof(int));
  LOG(INFO) << "row: "<<row<<"\ncol: "<<col<<"\ntype: "<<data_type;
  //std::cout << row << '\n' << col << '\n' << data_type;

  int frames_per_video = atoi(argv[4]);
  CHECK(col%frames_per_video == 0) << "col % frames_per_video !=0";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  int data_size;
  bool data_size_initialized = false;



  for (int line_id = 0; line_id < row; ++line_id) {
	  int label;
	  if_label_set >> label;
	  datum.set_channels(frames_per_video);
	  datum.set_height(col / frames_per_video);
	  datum.set_width(1);
	  datum.set_label(label);
	  datum.clear_data();
	  datum.clear_float_data();



	  for (int i = 0; i<col; i++)
	  {
		  float value;
		  if_feature_set.read((char*)&value, data_type);
		  datum.add_float_data(value);
	  }
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      //const string& data = datum.float_data();
      CHECK_EQ(datum.float_data().size(), data_size) << "Incorrect data field size "
          << datum.float_data().size();
    }
    // sequential
    _snprintf(key_cstr, kMaxKeyLength, "%08d", line_id);
    string value;
    // get the value
    datum.SerializeToString(&value);
    batch->Put(string(key_cstr), value);
    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR) << "Processed " << count << " files.";
  }

  delete batch;
  delete db;
  return 0;
}
