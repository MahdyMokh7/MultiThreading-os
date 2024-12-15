#include <iostream>
#include <sndfile.h>
#include <vector>
#include <string>
#include <cmath> 
#include <cstring>
#include <thread>
#include <chrono>
#include <functional>
#include <complex>
#include <iostream>
#include <vector>
#include <complex>
#include <pthread.h>
#include <thread>    
#include <chrono>
#include <algorithm>
#include <pthread.h>
#include <mutex>
#include "ReadWrite.h" 



#define INPUT_FOLDER_PATH "../"
#define OUTPUT_FOLDER_PATH "./"
#define NAME_INPUT "input.wav"
#define OUTPUT_NAME_BANDPASS "output_band_pass_Parallel.wav"
#define OUTPUT_NAME_NOTCH "output_notch_Parallel.wav"
#define OUTPUT_NAME_FIR "output_fir_Parallel.wav"
#define OUTPUT_NAME_IIR "output_iir_Parallel.wav"


using namespace std;


struct ThreadData {
    size_t startIdx;
    size_t endIdx;
    std::vector<std::complex<float>>& freqData;
    std::vector<float>& data;
    int sampleRate;
    float lowCut;
    float highCut;
    float deltaF;

    ThreadData(size_t start, size_t end,
               std::vector<std::complex<float>>& freqDataRef,
               std::vector<float>& dataRef,
               int sampleRate, float lowCut, float highCut, float deltaF)
        : startIdx(start), endIdx(end), freqData(freqDataRef), data(dataRef),
          sampleRate(sampleRate), lowCut(lowCut), highCut(highCut), deltaF(deltaF) {}

    ThreadData(const ThreadData&) = delete;
    ThreadData& operator=(const ThreadData&) = delete;
};

void* applyFilterChunk(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    size_t startIdx = data->startIdx;
    size_t endIdx = data->endIdx;
    auto& freqData = data->freqData;
    auto& originalData = data->data;
    int sampleRate = data->sampleRate;
    float lowCut = data->lowCut;
    float highCut = data->highCut;
    float deltaF = data->deltaF;

    for (size_t i = startIdx; i < endIdx; ++i) {
        float freq = (float)i * sampleRate / freqData.size();
        if (freq > sampleRate / 2) freq -= sampleRate; 

        float f2 = freq * freq;
        float Hf = (f2 / (f2 + deltaF * deltaF));  
        if (freq < lowCut || freq > highCut) {
            Hf = 0.0f;  
        }

        freqData[i] *= Hf;  

        originalData[i] = freqData[i].real();
    }

    return nullptr;
}

void applyBandPassFilter(std::vector<float>& data, int sampleRate, float lowCut, float highCut, int numThreads=4) {
    size_t n = data.size();
    std::vector<std::complex<float>> freqData(data.begin(), data.end());

    float deltaF = (highCut - lowCut) / 2.0f;

    pthread_t threads[numThreads];
    ThreadData* threadData[numThreads];

    size_t chunkSize = n / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = (i == numThreads - 1) ? n : (i + 1) * chunkSize;

        threadData[i] = new ThreadData(startIdx, endIdx, freqData, data, sampleRate, lowCut, highCut, deltaF);
        
        pthread_create(&threads[i], nullptr, applyFilterChunk, (void*)threadData[i]);
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
        delete threadData[i]; 
    }

    for (size_t i = 0; i < n; ++i) {
        data[i] = freqData[i].real();
    }
}





struct FilterData {
    const std::vector<float>* inputData;
    std::vector<float>* outputData;
    float norm_b0, norm_b1, norm_b2;
    float norm_a1, norm_a2;
    size_t startIdx, endIdx;
    float x1, x2, y1, y2; 
    std::mutex* stateMutex;
};

void applyNotchFilterChunk(FilterData* data) {
    auto& inputData = *data->inputData;
    auto& outputData = *data->outputData;
    float norm_b0 = data->norm_b0;
    float norm_b1 = data->norm_b1;
    float norm_b2 = data->norm_b2;
    float norm_a1 = data->norm_a1;
    float norm_a2 = data->norm_a2;

    float x1 = data->x1;
    float x2 = data->x2;
    float y1 = data->y1;
    float y2 = data->y2;

    for (size_t n = data->startIdx; n < data->endIdx; ++n) {
        float x = inputData[n];

        float y = norm_b0 * x + norm_b1 * x1 + norm_b2 * x2
                  - norm_a1 * y1 - norm_a2 * y2;

        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;

        outputData[n] = y;
    }

    std::lock_guard<std::mutex> lock(*data->stateMutex);
    data->x1 = x1;
    data->x2 = x2;
    data->y1 = y1;
    data->y2 = y2;
}

void applyNotchFilter(std::vector<float>& data, int samplerate, float notchFrequency, float bandwidth, int numThreads = 4) {
    const float dt = 1.0f / samplerate; 
    const float omega = 2.0f * M_PI * notchFrequency * dt;
    const float alpha = std::sin(omega) * std::sinh(std::log(2) / 2 * bandwidth * omega / std::sin(omega));

    const float b0 = 1.0f;
    const float b1 = -2.0f * std::cos(omega);
    const float b2 = 1.0f;
    const float a0 = 1.0f + alpha;
    const float a1 = -2.0f * std::cos(omega);
    const float a2 = 1.0f - alpha;

    const float norm_b0 = b0 / a0;
    const float norm_b1 = b1 / a0;
    const float norm_b2 = b2 / a0;
    const float norm_a1 = a1 / a0;
    const float norm_a2 = a2 / a0;

    std::vector<float> filteredData(data.size(), 0.0f);

    size_t chunkSize = data.size() / numThreads;
    std::vector<std::thread> threads;
    std::vector<FilterData> filterData(numThreads);
    std::mutex stateMutex;

    for (int t = 0; t < numThreads; ++t) {
        size_t startIdx = t * chunkSize;
        size_t endIdx = (t == numThreads - 1) ? data.size() : startIdx + chunkSize;

        filterData[t] = {
            &data, &filteredData,
            norm_b0, norm_b1, norm_b2, norm_a1, norm_a2,
            startIdx, endIdx,
            0.0f, 0.0f, 0.0f, 0.0f,
            &stateMutex
        };

        threads.emplace_back(applyNotchFilterChunk, &filterData[t]);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    data = filteredData;
}





struct FIRFilterThreadData {
    size_t startIdx;
    size_t endIdx;
    const std::vector<float>& data; 
    const std::vector<float>& coefficients; 
    std::vector<float>& filteredData;   

    FIRFilterThreadData(size_t start, size_t end, const std::vector<float>& dataRef,
                        const std::vector<float>& coeffRef, std::vector<float>& outputRef)
        : startIdx(start), endIdx(end), data(dataRef), coefficients(coeffRef), filteredData(outputRef) {}
};

void applyFIRFilterChunk(FIRFilterThreadData* threadData) {
    size_t startIdx = threadData->startIdx;
    size_t endIdx = threadData->endIdx;
    const auto& data = threadData->data;
    const auto& coefficients = threadData->coefficients;
    auto& filteredData = threadData->filteredData;

    size_t filterLength = coefficients.size();

    for (size_t n = startIdx; n < endIdx; ++n) {
        float outputSample = 0.0f;
        for (size_t k = 0; k < filterLength; ++k) {
            if (n >= k) {
                outputSample += coefficients[k] * data[n - k];
            }
        }
        filteredData[n] = outputSample;
    }
}

void applyFIRFilter(std::vector<float>& data, const std::vector<float>& coefficients, int numThreads = 4) {
    size_t dataSize = data.size();
    std::vector<float> filteredData(dataSize, 0.0f);

    std::vector<std::thread> threads;
    std::vector<FIRFilterThreadData*> threadData;

    size_t chunkSize = dataSize / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = (i == numThreads - 1) ? dataSize : (i + 1) * chunkSize;

        threadData.push_back(new FIRFilterThreadData(startIdx, endIdx, data, coefficients, filteredData));
        threads.emplace_back(applyFIRFilterChunk, threadData.back());
    }

    for (auto& thread : threads) {
        thread.join();
    }

    for (auto* td : threadData) {
        delete td;
    }

    data = filteredData;
}





void applyIIRFilterChunk(std::vector<float>& data, float a0, float a1, float b1, 
                         size_t startIdx, size_t endIdx, float& prevInput, float& prevOutput) {
    // Temporary variables for local recursion within the chunk
    float localPrevInput = prevInput;
    float localPrevOutput = prevOutput;

    for (size_t i = startIdx; i < endIdx; ++i) {
        float currentSample = data[i];
        float output = a0 * currentSample + a1 * localPrevInput - b1 * localPrevOutput;

        localPrevInput = currentSample;
        localPrevOutput = output;

        data[i] = output;
    }

    // Update the shared recursion states
    prevInput = localPrevInput;
    prevOutput = localPrevOutput;
}

void applyIIRFilter(std::vector<float>& data, float a0, float a1, float b1, int numThreads) {
    size_t dataSize = data.size();
    if (dataSize == 0 || numThreads < 1) return;

    // Divide the data into chunks
    size_t chunkSize = (dataSize + numThreads - 1) / numThreads; // Round up

    // Create threads
    std::vector<std::thread> threads(numThreads);
    std::vector<float> prevInputs(numThreads, 0.0f);
    std::vector<float> prevOutputs(numThreads, 0.0f);

    for (int t = 0; t < numThreads; ++t) {
        size_t startIdx = t * chunkSize;
        size_t endIdx = std::min(dataSize, startIdx + chunkSize);

        if (startIdx >= dataSize) break; // If there are more threads than chunks

        threads[t] = std::thread(applyIIRFilterChunk, 
                                 std::ref(data), a0, a1, b1, 
                                 startIdx, endIdx, 
                                 std::ref(prevInputs[t]), 
                                 std::ref(prevOutputs[t]));
    }

    // Join threads
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Resolve boundary dependencies between chunks
    for (int t = 1; t < numThreads; ++t) {
        size_t boundaryIdx = t * chunkSize;
        if (boundaryIdx < dataSize) {
            // Update the boundary values to maintain correctness
            data[boundaryIdx] += a1 * prevInputs[t - 1] - b1 * prevOutputs[t - 1];
        }
    }
}


void measureTime(const function<void()>& task, string filter_name) {
    auto start = std::chrono::high_resolution_clock::now();

    task();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cout << filter_name << "  Time taken: " << duration.count() << " milliseconds" << std::endl;
}

void printFileInfo(const string& fileName, const SF_INFO& fileInfo, size_t dataSize) {
    cout << "File: " << fileName << endl;
    cout << "Sample Rate: " << fileInfo.samplerate << " Hz" << endl;
    cout << "Channels: " << fileInfo.channels << endl;
    cout << "Frames: " << fileInfo.frames << endl;
    cout << "Data Size: " << dataSize << " samples" << endl;
    cout << "----------------------------------" << endl;
}


int main(int argc, char* argv[]) {

    auto start = std::chrono::high_resolution_clock::now();
    
    SF_INFO fileInfo;
    vector<float> data_voice;
    vector<float> data_voice_bandpass;
    vector<float> data_voice_notch;
    vector<float> data_voice_fir;
    vector<float> data_voice_iir;

    const string input_name = argv[1];
    const string input_folder_path = INPUT_FOLDER_PATH;
    const string output_folder_path = OUTPUT_FOLDER_PATH;
    const string output_name_band_pass = OUTPUT_NAME_BANDPASS;
    const string output_name_notch = OUTPUT_NAME_NOTCH;
    const string output_name_fir = OUTPUT_NAME_FIR;
    const string output_name_iir = OUTPUT_NAME_IIR;

    int samplerate = 44100;

    const string input_File_path = input_folder_path + input_name;
    const string output_bandpass_File_path = output_folder_path + output_name_band_pass;
    const string output_notch_File_path = output_folder_path + output_name_notch;
    const string output_fir_File_path = output_folder_path + output_name_fir;
    const string output_iir_File_path = output_folder_path + output_name_iir;

    // Read the input.wav file
    memset(&fileInfo, 0, sizeof(fileInfo));
    measureTime([&]() {
        readWavFile(input_File_path, data_voice, fileInfo);
    }, "readWavFile");
    printFileInfo(input_File_path, fileInfo, data_voice.size());

    
    // Initialize vectors for filtered data
    data_voice_bandpass = std::vector<float>(data_voice.begin(), data_voice.end());
    data_voice_notch = std::vector<float>(data_voice.begin(), data_voice.end());
    data_voice_fir = std::vector<float>(data_voice.begin(), data_voice.end());
    data_voice_iir = std::vector<float>(data_voice.begin(), data_voice.end());

    int numThreads = std::thread::hardware_concurrency();
    cout << "++++++++++++++++++++++numthreads:  " << numThreads << endl;


    // BandPass filter
    float low_pass_human = 20.0f;
    float high_pass_human = 20000.0f;
    measureTime([&]() {
        applyBandPassFilter(data_voice_bandpass, samplerate, low_pass_human, high_pass_human, numThreads);
    }, "BandPass filter");
    writeWavFile(output_bandpass_File_path, data_voice_bandpass, fileInfo);
    printFileInfo(output_bandpass_File_path, fileInfo, data_voice_bandpass.size());

    // Notch filter
    float notchFrequency = 55.0f;
    float bandwidth = 5.0f;
    measureTime([&]() {
        applyNotchFilter(data_voice_notch, fileInfo.samplerate, notchFrequency, bandwidth, numThreads);
    }, "Notch Filter");
    writeWavFile(output_notch_File_path, data_voice_notch, fileInfo);  // Corrected to use data_voice_notch
    printFileInfo(output_notch_File_path, fileInfo, data_voice_notch.size());

    // FIR filter
    vector<float> firCoefficients = {
        0.1f, 0.15f, 0.5f, 0.15f, 0.1f
    };
    measureTime([&]() {
        applyFIRFilter(data_voice_fir, firCoefficients, numThreads);
    }, "FIR Filter");
    writeWavFile(output_fir_File_path, data_voice_fir, fileInfo);
    printFileInfo(output_fir_File_path, fileInfo, data_voice_fir.size());


    // IIR filter
    float a0 = 0.1f; 
    float a1 = 0.2f; 
    float b1 = 0.3f; 
    measureTime([&]() {
        applyIIRFilter(data_voice_iir, a0, a1, b1, numThreads);
    }, "IIR Filter");
    writeWavFile(output_iir_File_path, data_voice_iir, fileInfo);
    printFileInfo(output_iir_File_path, fileInfo, data_voice_iir.size());


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count()*1000 << " MiliSeconds" << std::endl;
}
