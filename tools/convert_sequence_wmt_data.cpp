#include<fstream>
#include<string>
#include<time.h>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "caffe/util/db.hpp"
#include "stdint.h"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe\util\math_functions.hpp"
#include "boost\scoped_ptr.hpp"
#include "opencv2\core\core.hpp"
#include "boost/algorithm/string.hpp"

using namespace std;
using namespace caffe;
using boost::scoped_ptr;
using std::string;

DEFINE_int32(feat_dim, 160001, "The # of words in the word set");
DEFINE_string(backend, "lmdb", "The backend{leveldb/lmdb} for storing the result");
DEFINE_bool(train, true, "if the dataset is for train(true: train, false: test)");

void parse_line_feat(string& line, vector<int>& feat){
	feat.clear();
	vector<string> strs;
	//delete spaces in the beginning and ending of the sequence
	boost::trim(line);
	boost::split(strs, line, boost::is_any_of(" "));
	int feat_i;
	for (vector<string>::iterator it = strs.begin(); 
		it != strs.end(); ++it){
		feat_i = stoi(*it);
		feat.push_back(feat_i);
	}
}

void create_db(const string& feat_file, const string& db_name){
	//load feat file
	ifstream in_feat(feat_file);
	CHECK(in_feat.is_open()) << "open " << feat_file << " failed!";

	string test_line;
	getline(in_feat, test_line);
	vector<int> feats;
	parse_line_feat(test_line, feats);

	//create new db
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(db_name, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];

	//save image data into leveldb
	int count = 0;
	Datum sentence_datum;
	const int dim = FLAGS_train ? FLAGS_feat_dim : (FLAGS_feat_dim + 2);
	LOG(INFO) << "dim: " << dim;
	/*
	 *datum: channels, width, height, encoded, data(), float_data()
	 *export: datum.SerializeToString(&out)
	 */
	sentence_datum.set_height(1);
	// sentence_datum.dim = feat_dim + 1(begin mark) + 1(end mark)
	sentence_datum.set_width(dim);

	//build training dataset
	int num_sentences = 0;
	const int label = 1;
	string line;
	vector<int> feat_ids;
	while (getline(in_feat, line)){
		parse_line_feat(line, feat_ids);
		//for targeting sequence, we need to add start mark and end 
		//mark
		if (!FLAGS_train){
			//begin mark
			feat_ids.insert(feat_ids.begin(), FLAGS_feat_dim + 1);
			//end mark
			feat_ids.push_back(FLAGS_feat_dim + 2);
		}
		sentence_datum.set_channels((int)feat_ids.size());
		sentence_datum.clear_float_data();
		for (int i = 0; i < feat_ids.size(); i++){
//			cout << feat_ids[i] << "\t";
			for (int f = 0; f < dim; f++){
				if (f == feat_ids[i]){
					sentence_datum.add_float_data(1);
				}
				else{
					sentence_datum.add_float_data(0);
				}
			}
		}
		//sequential 
		string out;
		sentence_datum.SerializeToString(&out);
//		LOG(INFO)<< "feat_id size: " << feat_ids.size();
//		LOG(INFO) << "float data size: " << sentence_datum.float_data_size();
		int length = sprintf_s(key_cstr, kMaxKeyLength, "%09d", count);
		//put into db
		txn->Put(std::string(key_cstr, length), out);
		num_sentences++;
		if (++count % 1000 == 0){
			//commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << num_sentences << " sentences";
		}
	}//while getline
	//write the last batch
	if (count % 1000 != 0){
		txn->Commit();
		LOG(INFO) << "Processed " << count << " sentences";
	}
}

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::SetStderrLogging(0);
	if (argc < 3){
		gflags::SetUsageMessage("Convert a set of pair images to leveldb\n"
			"format used as input for caffe.\n"
			"usage: \n"
			" EXE [FLAGS] INPUT_FEAT_FILE DB_NAME\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_sequence_wmt_data");
		return 1;
	}

	create_db(argv[1], argv[2]);
	return 0;
}