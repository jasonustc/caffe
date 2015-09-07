#include <cutil_inline.h>

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
	int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
	int moduleStride, int numImgColors, int numGroups);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
	int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
	int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
	int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
	int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
	float scaleTargets, float scaleOutput);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
	int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
	int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
	float scaleTargets, float scaleOutput);
