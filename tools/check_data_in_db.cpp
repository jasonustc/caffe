#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/db.hpp"
#include "opencv2\opencv.hpp"
#include "boost/scoped_ptr.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;
using boost::scoped_ptr;
using namespace caffe;

DEFINE_string(backend, "lmdb", "the backend{lmdb, leveldb} for storing the result.");
DEFINE_int32(num, 10, "# of samples to read in and write to output file.");

int main(int argc, char** argv){
	::google::InitGoogleLogging(argv[0]);
	::google::SetStderrLogging(0);
	::gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc != 3){
		::gflags::SetUsageMessage("Usage: EXE input_db output_db_data_file");
		::gflags::ShowUsageWithFlags(argv[0]);
		return 1;
	}

	FILE* output = fopen(argv[2], "w");
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[1], db::READ);
	shared_ptr<db::Cursor> cursor;
	cursor.reset(db->NewCursor());

	Datum datum;
	int count = 0;

	LOG(INFO) << "Start Iteration...";
	cv::Mat cv_img;
	for (cursor->SeekToFirst(); cursor->valid(); cursor->Next()){
		//just a dummy operation
		datum.ParseFromString(cursor->value());
		const std::string& data = datum.data();
		const int size = datum.encoded() ? datum.data().size() : datum.float_data_size();
		if (count == 0){
			LOG(INFO)<< "sample size info:";
			LOG(INFO) << "channels: " << datum.channels();
			LOG(INFO) << "width: " << datum.width();
			LOG(INFO) << "height: " << datum.height();
		}
		CHECK_EQ(size, datum.channels() * datum.width() * datum.height());
		if (count == 0){
			fprintf(output, "[C, W, H]: %d %d %d\n", datum.channels(), datum.width(), datum.height());
		}

		//default type of encoded in datum is false
		if (datum.encoded()){
			//if encoded, data is coded from opencv mat
			//and stored in data
			//so we need to decode to cv::MAT first
			cv_img = DecodeDatumToCVMatNative(datum);
			const int channels = cv_img.channels();
			switch (channels)
			{
			case 1:
			{
			  cv::MatIterator_<uchar> it, end;
			  for (it = cv_img.begin<uchar>(); end != cv_img.end<uchar>(); ++it){
				  fprintf(output, "%d ", static_cast<unsigned int>(*it));
			  }
			  break;
			}
			case 3:
			{
			  cv::MatIterator_<cv::Vec3b> it, end;
			  for (it = cv_img.begin<cv::Vec3b>(); end != cv_img.end<cv::Vec3b>(); ++it){
				  fprintf(output, "%d ", static_cast<unsigned int>((*it)[0]));
				  fprintf(output, "%d ", static_cast<unsigned int>((*it)[1]));
				  fprintf(output, "%d ", static_cast<unsigned int>((*it)[2]));
			  }
			  break;
			}
			default:
				LOG(FATAL) << "decoded cv image must have 1/3 channels";
				break;
			}
		}
		else{
			//if not encoded, data is stored in float data
			for (int i = 0; i < size; i++){
				fprintf(output, "%f ", static_cast<float>(datum.float_data(i)));
			}
		}
		++count;
		if (count > FLAGS_num){
			break;
		}
		if (count % 1000 == 0){
			LOG(ERROR) << "Have read: " << count << " files.";
		}
		fprintf(output, "\n");
	}
	if (count % 1000 != 0){
		LOG(INFO) << "Processed " << count << " files.";
	}
	fclose(output);
	return 0;
}