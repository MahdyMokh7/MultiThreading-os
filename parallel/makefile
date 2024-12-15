CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
LDFLAGS = -lsndfile

SRCS = main_parallel.cpp ReadWrite.cpp
OBJS = $(SRCS:.cpp=.o)

all: parallel.out

parallel.out: main_parallel.o ReadWrite.o
	$(CXX) $(CXXFLAGS) -o $@ main_parallel.o ReadWrite.o $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) parallel.out
	find . -maxdepth 1 -type f -name "*.wav" -exec rm -f {} \;
