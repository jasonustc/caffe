#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include "caffe/fea_extractor.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace std;

DEFINE_int32(GPU, -1, "if positive, we use GPU with given id, else use CPU instead.");
DEFINE_bool(multi_crop, false, "if we test by 10 crops or center crop only.");

void load_file_list(const string& file_path, vector<string>& imgList){
	imgList.clear();
	ifstream inImages(file_path);
	if (!inImages.is_open()){
		LOG(FATAL) << "can not load indexes from file " << file_path;
	}
	string imgPath;
	while (inImages >> imgPath){
		imgList.push_back(imgPath);
	}
	LOG(INFO) << "Load " << imgList.size() << " images.";
}

void load_file_list(const string& file_path, vector<pair<string, int>>& imgList,
	string folder_path){
	imgList.clear();
	ifstream inImages(file_path);
	if (!inImages.is_open()){
		LOG(FATAL) << "can not load indexes from file " << file_path;
	}
	string imgPath;
	int label;
	string abs_img_path;  
	while (inImages >> imgPath >> label){
		abs_img_path = folder_path + "\\" + imgPath;
		string abs_img_path = folder_path + "\\" + imgPath;
		imgList.push_back(std::make_pair(abs_img_path, label));
	}
	LOG(INFO) << "Load " << imgList.size() << " images.";
}

bool is_txt(const string& file_path){
	string ext = file_path.substr(file_path.rfind('.') + 1);
	return ext == "txt";
}

bool load_coeffs(const string& file_path,float* coeffs, const int dim = 4096){
	ifstream inCoeff(file_path.c_str());
	CHECK(inCoeff.is_open()) << "Failed to open coefficients file " << file_path.c_str();
	int i = 0;
	float feat;
	while (inCoeff >> feat){
		coeffs[i] = feat;
		i++;
	}
	CHECK_EQ(i, dim) << "Feature dim not match with #coeffs in file " << file_path.c_str();
	inCoeff.close();
	return true;
}

void NormalizeFeat(const int count, float* feat, const float* coeffs){
	//divided by max value in each dim
	caffe::caffe_div<float>(count, feat, coeffs, feat);
	//sqare
	caffe::caffe_sqr(count, feat, feat);
	//sum of square
	float sqr_sum = caffe::caffe_cpu_asum(count, feat);
	//normalize
	caffe::caffe_scal<float>(count, (float)1. / sqr_sum, feat);
}

bool isTruePred(const int count, const float* feat, const int label, const int top_k){
	std::vector<std::pair<float, int>> feat_data;
	for (int i = 0; i < count; i++){
		feat_data.push_back(
			std::make_pair(feat[i], i));
	}
	std::partial_sort(feat_data.begin(), feat_data.begin() + top_k,
		feat_data.end(), std::greater<std::pair<float, int>>());
	//check if true label is in the top k predictions
	for (int k = 0; k < top_k; k++){
		if (label == feat_data[k].second){
			return true;
		}
	}
	return false;
}

void Average(const int dim, const int count, const float* a, const float* b, float* dst){
	for (int i = 0; i < dim; i++){
		dst[i] = (a[i]* count + b[i]) / (count + 1);
	}
}

int main(int argc, char** argv) {
	google::InitGoogleLogging(*argv);
	google::SetStderrLogging(0);
	if (argc < 8){
		LOG(FATAL) << "Usage: test_multi_crop [FLAGS] model_path net_path "
			<< "mean_path layer_name img_index_path(txt file) top_k";
		gflags::ShowUsageWithFlags(*argv);
		return 0;
	}

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	string model_path = argv[1];
	string net_path = argv[2];
	string mean_path = argv[3];
	string layer_name = argv[4];
	string img_path = argv[5];
	string folder_path = argv[6];
	int top_k = stoi(argv[7]);
	vector<std::pair<string, int>> img_list;
	bool txt = is_txt(img_path);
	if (txt){
		load_file_list(img_path, img_list, folder_path);
		CHECK_GT(img_list.size(), 0) << "no images in the txt file";
	}
	

	FeaExtractor<float>* fea_extractor = NULL;
	bool succ = CreateHandle(fea_extractor);
	if (!succ)
	{
		cout << "error, can not create handle" << endl;
		return -1;
	}

	if (FLAGS_GPU >= 0){
		succ = LoadDModel(fea_extractor, mean_path, net_path, 
			model_path, layer_name, Caffe::GPU, FLAGS_GPU); // load vgg model
	}
	else{
		succ = LoadDModel(fea_extractor, mean_path, 
			net_path, model_path, layer_name); // load vgg model
	}
	if (!succ)
	{
		cout << "extractor failed to load model" << endl;
		return -1;
	}

	int dim = GetFeaDim(fea_extractor); // get the dimension of features
	float* coeffs = new float[dim];
	int channel = 3;

	int numTruePred = 0;
	int count = 0;
	int num_processed = 0;
	for (size_t i = 0; i < img_list.size(); i++)
	{
		const float* fea;
		//average prediction by 10 crops:
		//mirror, not mirror
		//left up, right up, middle, left down, right down
		if (FLAGS_multi_crop){
			ExtractCropFeaturesByPath(
				fea_extractor,
				img_list[i].first, dim);
			fea = GetFeaPtr(fea_extractor);
		}
		else{
			ExtractFeaturesByPath(
				fea_extractor,
				img_list[i].first);
			fea = GetFeaPtr(fea_extractor);
		}

		bool is_true = isTruePred(dim, fea, img_list[i].second, top_k);
		num_processed++;
		if (is_true){
			numTruePred++;
		}
		if (num_processed % 100 == 0){
			LOG(INFO) << "Accuracy: " << float(numTruePred) / float(num_processed);
		}
	}
	float accuracy = float(numTruePred) / float(img_list.size());
	printf("accuracy: %f", accuracy);

	ReleaseHandle(fea_extractor);
	return 0;
}