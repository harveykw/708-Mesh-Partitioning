# Makefile for 2D Mesh Partitioning Program

CXX = mpicxx
CXXFLAGS = -std=c++17 -O2 -Wall

TARGET = partitioner
SRCS = main.cpp mesh.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp mesh.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o partitions.txt mesh.bin

.PHONY: all clean
