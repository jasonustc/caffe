#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "boost/algorithm/string.hpp"

using caffe::Datum;
using namespace std;
using namespace caffe;

DEFINE_int32(channels, 1, "channels of the image");
DEFINE_int32(height, 28, "height of the image");
DEFINE_int32(width, 28, "width of the image");
DEFINE_string(backend, "lmdb", "The backend{leveldb/lmdb} for storing the result");
DEFINE_int32(db_size, INT_MAX, "number of samples in each db");

void parse_line_feat(string& line, vector<float>& feat){
	feat.clear();
	vector<string> strs;
	//delete spaces in the beginning and ending of the sequence
	boost::trim(line);
	boost::split(strs, line, boost::is_any_of(" "));
	float feat_i;
	for (vector<string>::iterator it = strs.begin(); 
		it != strs.end(); ++it){
		if ((*it).size() == 0){
			continue;
		}
		istringstream iss(*it);
		iss >> feat_i;
		feat.push_back(feat_i);
	}
}

void convert_dataset_float (const string& feat_file, const string& label_file,
	const string& db_name) {
	vector<string> feat_files;
	vector<string> label_files;
	vector<string> db_names;
	boost::split(feat_files, feat_file, boost::is_any_of(","));
	boost::split(label_files, label_file, boost::is_any_of(","));
	boost::split(db_names, db_name, boost::is_any_of(","));
	CHECK_EQ(feat_files.size(), label_files.size()) << "number of feat_files and "
		<< "label files should be the same";

	//create db
	vector<boost::shared_ptr<db::Transaction>> txns;
	vector<boost::shared_ptr<db::DB>> dbs;
	for (size_t d = 0; d < db_names.size(); d++){
		boost::shared_ptr<db::DB> db(db::GetDB(FLAGS_backend));
		db->Open(db_names[d], db::NEW);
		boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
		txns.push_back(txn);
		dbs.push_back(db);
	}

	// Data buffer
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	Datum datum;

	CHECK(FLAGS_channels > 0 && FLAGS_height > 0 && FLAGS_width > 0)
		<< "channels, height and width should be positive; while it is set to be "
		<< FLAGS_channels << "," << FLAGS_height << "," << FLAGS_width;
	datum.set_channels(FLAGS_channels);
	datum.set_height(FLAGS_height);
	datum.set_width(FLAGS_width);

	LOG(ERROR) << "Loading data";
	string line;
	vector<float> feats;
	int label = -1;
	int count = 0;
	int db_index = 0;
	for (size_t f = 0; f < feat_files.size(); f++){
		printf("Loading feat from %s, label from %s\n", 
			feat_files[f].c_str(), label_files[f].c_str());
		ifstream in_feat(feat_files[f]);
		CHECK(in_feat.is_open()) << "Can not open feat file: " << feat_files[f];
		//	string test_line;
		//	getline(in_feat, test_line);
		//	vector<float> test_feat;
		//	parse_line_feat(test_line, test_feat);
		ifstream in_label(label_files[f]);
		CHECK(in_label.is_open()) << "Can not open label file: " << label_files[f];
		while (getline(in_feat, line)){
			in_label >> label;
			CHECK_GE(label, 0);
			datum.set_label(label);
			parse_line_feat(line, feats);
			//check feat dim 
			int feat_dim = FLAGS_channels * FLAGS_height * FLAGS_width;
			CHECK_EQ(feat_dim, feats.size()) << "Feat dim not match, required: "
				<< feat_dim << ", get: " << feats.size();
			//save feat/score data to db
			datum.clear_float_data();
			for (int i = 0; i < feats.size(); i++){
				datum.add_float_data(feats[i]);
			}
			//sequential
			string out;
			datum.SerializeToString(&out);
			int len = sprintf_s(key_cstr, kMaxKeyLength, "%09d", count);
			db_index = count / FLAGS_db_size;
			CHECK_LT(db_index, db_names.size());
			//put into db
			txns[db_index]->Put(std::string(key_cstr, len), out);
			if ((++count) % 1000 == 0){
				//commit db
				txns[db_index]->Commit();
				txns[db_index].reset(dbs[db_index]->NewTransaction());
				LOG(ERROR) << "Processed " << count << " feats" <<
					", DB: " << db_names[db_index];
			}
		}//while getline
		if (count % 1000 != 0){
			txns[db_index]->Commit();
			LOG(ERROR) << "Processed " << count << " feats" <<
				", DB: " << db_names[db_index];
		}
	}//for (size_t f = 0; f < feat_files.size(); f++)
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);
	//parse flags
	::gflags::ParseCommandLineFlags(&argc, &argv, true);
	if (argc < 4) {
		gflags::SetUsageMessage("Convert feats and scores by line to lmdb/leveldb\n"
			"format used for caffe.\n"
			"Usage: \n"
			"EXE [FLAGS] INPUT_FEAT_FILEs(,) INPUT_LABEL_FILEs(,) DB_NAMEs(,)\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_feat_label_data");
		return 1;
	}
	else {
		LOG(INFO) << "Channels: " << FLAGS_channels << ", height: " 
			<< FLAGS_height << ", width: " << FLAGS_width;
		convert_dataset_float(string(argv[1]), string(argv[2]), string(argv[3]));
	}
	return 0;
}
