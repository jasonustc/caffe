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

void parse_line_feat(string& line, vector<float>& feat){
	feat.clear();
	vector<string> strs;
	//delete spaces in the beginning and ending of the sequence
	boost::trim(line);
	boost::split(strs, line, boost::is_any_of(" "));
	float feat_i;
	for (vector<string>::iterator it = strs.begin(); 
		it != strs.end(); ++it){
		feat_i = stof(*it);
		feat.push_back(feat_i);
	}
}

void create_db(const string& feat_file, const string& db_name){
	//load feat file
	ifstream in_feat(feat_file);
	CHECK(in_feat.is_open()) << "open " << feat_file << " failed!";

	/*
	string test_line;
	getline(in_feat, test_line);
	vector<float> test_feats;
	parse_line_feat(test_line, test_feats);
	*/

	//create new db
	string feat_db_name = db_name + "_feat";
	string score_db_name = db_name + "_score";
	scoped_ptr<db::DB> feat_db(db::GetDB(FLAGS_backend));
	feat_db->Open(feat_db_name, db::NEW);
	scoped_ptr<db::Transaction> txn_feat(feat_db->NewTransaction());
	scoped_ptr<db::DB> score_db(db::GetDB(FLAGS_backend));
	score_db->Open(score_db_name, db::NEW);
	scoped_ptr<db::Transaction> txn_score(score_db->NewTransaction());

	const int kMaxKeyLength = 256;
	char key_cstr_feat[kMaxKeyLength];
	char key_cstr_score[kMaxKeyLength];

	//save image data into leveldb
	int count = 0;
	Datum feat_datum;
	Datum score_datum;
	/*
	 *datum: channels, width, height, encoded, data(), float_data()
	 *export: datum.SerializeToString(&out)
	 */
	feat_datum.set_height(1);
	feat_datum.set_width(2);
	score_datum.set_channels(1);
	score_datum.set_height(1);
	score_datum.set_width(1);

	//build training dataset
	int num_sentences = 0;
	const int label = 1;
	string line;
	vector<float> feats;
	while (getline(in_feat, line)){
		parse_line_feat(line, feats);
		//save feat/score data to db
		//the feat of each data has two dim: data and mask
		LOG(ERROR) << "feats size: " << feats.size();
		feat_datum.set_channels((int)feats.size()/2);
		LOG(ERROR) << "channels: " << feat_datum.channels();
		feat_datum.clear_float_data();
		for (int i = 0; i < feats.size() - 1; i++){
			if (feats[i] == 1){
				printf("%f, %f\n", feats[i - 1], feats[i]);
			}
			feat_datum.add_float_data(feats[i]);
		}
		score_datum.clear_float_data();
		score_datum.add_float_data(feats[feats.size() - 1]);
		printf("%f \n", feats[feats.size() - 1]);
		//sequential 
		string out_feat;
		string out_score;
		feat_datum.SerializeToString(&out_feat);
		score_datum.SerializeToString(&out_score);
		int length_feat = sprintf_s(key_cstr_feat, kMaxKeyLength, "%09d", count);
		int length_score = sprintf_s(key_cstr_score, kMaxKeyLength, "%09d", count);
		//put into db
		txn_feat->Put(std::string(key_cstr_feat, length_feat), out_feat);
		txn_score->Put(std::string(key_cstr_score, length_feat), out_score);
		if (++count % 1000 == 0){
			//commit db
			txn_feat->Commit();
			txn_score->Commit();
			txn_feat.reset(feat_db->NewTransaction());
			txn_score.reset(score_db->NewTransaction());
			LOG(INFO) << "Processed " << count << " sequences";
		}
	}//while getline
	//write the last batch
	if (count % 1000 != 0){
		txn_feat->Commit();
		txn_score->Commit();
		LOG(INFO) << "Processed " << count << " sequences";
	}
}

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::SetStderrLogging(0);
	if (argc < 3){
		gflags::SetUsageMessage("Convert a set of feats(by line) to lmdb/leveldb\n"
			"format used as input for caffe.\n"
			"usage: \n"
			" EXE [FLAGS] INPUT_FEAT_FILE DB_NAME\n");
		gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_sequence_adding_data");
		return 1;
	}

	create_db(argv[1], argv[2]);
	return 0;
}