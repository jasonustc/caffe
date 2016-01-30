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

DEFINE_string(backend, "lmdb", "The backend{leveldb/lmdb} for storing the result");
DEFINE_int32(sequence_length, 25, "The length of each sequence");
DEFINE_int32(feat_dim, 4096, "the dimension of the feature");
DEFINE_string(splitor, " ", "The string used to segment each dim of feature in the line");
DEFINE_int32(split_loc, 4631, "The location index of sentence to split train and test");
DEFINE_bool(shuffle, true, "if we need to shuffle the order of samples");

void parse_line_feat(string& line, vector<float>& feat){
	feat.clear();
	vector<string> strs;
	//delete spaces in the beginning and ending of the sequence
	boost::trim(line);
	boost::split(strs, line, boost::is_any_of(FLAGS_splitor));
	float feat_i;
	for (vector<string>::iterator it = strs.begin(); 
		it != strs.end(); ++it){
		if ((*it).length()){
			feat_i = stof(*it);
			feat.push_back(feat_i);
		}
	}
}

void create_db(const string& feat_file, const string& label_file, const string& db_name){
	//load feat file
	ifstream in_feat(feat_file);
	CHECK(in_feat.is_open()) << "open " << feat_file << " failed!";
	ifstream in_label(label_file);
	CHECK(in_label.is_open()) << "open " << label_file << " failed!";

	/*
	string test_line;
	getline(in_feat, test_line);
	vector<float> test_feats;
	parse_line_feat(test_line, test_feats);
	*/

	//create new db
	string train_feat_db_name = db_name + "_train";
	scoped_ptr<db::DB> train_feat_db(db::GetDB(FLAGS_backend));
	train_feat_db->Open(train_feat_db_name, db::NEW);
	scoped_ptr<db::Transaction> txn_train_feat(train_feat_db->NewTransaction());

	string test_feat_db_name = db_name + "_test";
	scoped_ptr<db::DB> test_feat_db(db::GetDB(FLAGS_backend));
	test_feat_db->Open(test_feat_db_name, db::NEW);
	scoped_ptr<db::Transaction> txn_test_feat(test_feat_db->NewTransaction());

	const int kMaxKeyLength = 256;
	char key_cstr_feat[kMaxKeyLength];
	char key_cstr_score[kMaxKeyLength];

	//save image data into leveldb
	Datum feat_datum;
	/*
	 *datum: channels, width, height, encoded, data(), float_data()
	 *export: datum.SerializeToString(&out)
	 */
	feat_datum.set_height(1);
	feat_datum.set_width(FLAGS_feat_dim);
	feat_datum.set_channels(FLAGS_sequence_length);

	//build training dataset
	int label = 1;
	string line, seq_line, label_str;
	vector<float> feats;
	int line_count = 0;
	int count = 0;
	vector<string> lines;
	while (getline(in_feat, line)){
		line_count++;
		seq_line += line;
		if (line_count % FLAGS_sequence_length == 0){
			//TODO: check why the number of features are not 
			//equal to the number of labels
			if (getline(in_label, label_str)){
				seq_line += label_str;
				lines.push_back(seq_line);
				//clear 
				seq_line = "";
			}
			else{
				break;
			}
		}
	}
	LOG(INFO) << lines.size();
//	CHECK_EQ(lines.size(), line_count / FLAGS_sequence_length);
	if (FLAGS_shuffle){
		std::random_shuffle(lines.begin(), lines.end());
	}
	for (size_t line_id = 0; line_id < lines.size(); line_id++){
		line = lines[line_id];
		parse_line_feat(line, feats);
		//save feat/score data to db
		//the feat of each data has two dim: data and mask
		if (count == 0){
			LOG(ERROR) << "feats size: " << feats.size();
		}
		CHECK_EQ(feats.size(), FLAGS_feat_dim * FLAGS_sequence_length + 1) << 
			"One line per sequence feature + label";
		count++;
		feat_datum.clear_float_data();
		for (int i = 0; i < feats.size() - 1; i++){
			feat_datum.add_float_data(feats[i]);
		}
		int label = feats[feats.size() - 1];
		LOG(ERROR) << label;
		feat_datum.set_label(label);
		//sequential 
		string out_feat;
		CHECK_EQ(feat_datum.float_data_size(), FLAGS_feat_dim * FLAGS_sequence_length);
		feat_datum.SerializeToString(&out_feat);
		int length_feat = sprintf_s(key_cstr_feat, kMaxKeyLength, "%09d", count);
		//put into db
		if (count < (FLAGS_split_loc + 1)){
			txn_train_feat->Put(std::string(key_cstr_feat, length_feat), out_feat);
			//commit db
			txn_train_feat->Commit();
			txn_train_feat.reset(train_feat_db->NewTransaction());
			if (count % 1000 == 0){
				LOG(ERROR) << "Processed " << count << " train sequences";
			}
		}
		else{
			txn_test_feat->Put(std::string(key_cstr_feat, length_feat), out_feat);
			//commit db
			txn_test_feat->Commit();
			txn_test_feat.reset(test_feat_db->NewTransaction());
			if (count % 1000 == 0){
				LOG(ERROR) << "Processed " << (count - FLAGS_split_loc) << " test sequences";
			}
		}
	}// for line_id
	if (count % 1000 != 0){
		LOG(ERROR) << "Processed " << count << " Sequences";
	}
	//write the last batch
	/*
	if (count % 1000 != 0){
		txn_feat->Commit();
		LOG(INFO) << "Processed " << count << " sequences";
	}
	*/
}

/*
 * each line stores all the feature of a sequence, segmented by splitor
 * the feautre should be in float format
 */

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::SetStderrLogging(0);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	if (argc < 4){
		gflags::SetUsageMessage("Convert a set of feats(by line) to lmdb/leveldb\n"
			"format used as input for caffe.\n"
			"usage: \n"
			" EXE [FLAGS] INPUT_FEAT_FILE INPUT_LABEL_FILE DB_NAME\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_sequence_feature");
		return 1;
	}

	create_db(argv[1], argv[2], argv[3]);
	return 0;
}