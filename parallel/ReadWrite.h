#ifndef READWRITE_H
#define READWRITE_H

#include <iostream>
#include <sndfile.h>
#include <vector>
#include <string>
#include <cstring>

using namespace std;
void readWavFile(const std::string& inputFile, std::vector<float>& data, SF_INFO& fileInfo);

void writeWavFile(const std::string& outputFile, const std::vector<float>& data, SF_INFO fileInfo);



#endif 