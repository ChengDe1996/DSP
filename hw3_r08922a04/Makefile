SRIPATH ?= /root/srilm-1.5.10
MACHINE_TYPE ?= i686-m64
INC_PATH ?= inc
SRC_PATH ?= src

CXX = g++
CXXFLAGS = -O2 -I$(SRIPATH)/include -I$(INC_PATH) -W
vpath lib%.a $(SRIPATH)/lib/$(MACHINE_TYPE)
vpath %.c $(SRC_PATH)
vpath %.cpp $(SRC_PATH)

TARGET = mydisambig
TARGET_TRIGRAM = mydisambig_trigram
SRC = mydisambig.cpp
SRC_TRIGRAM = mydisambig_trigram.cpp
OBJ = $(SRC:.cpp=.o)
OBJ_TRIGRAM = $(SRC_TRIGRAM:.cpp=.o)

FROM ?= Big5-ZhuYin.map
TO ?= ZhuYin-Big5.map
.PHONY: all clean map

$(TARGET): $(OBJ) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^

$(TARGET_TRIGRAM): $(OBJ_TRIGRAM) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<


all: $(TARGET_TRIGRAM) $(TARGET)


run:
	@#TODO run my code for all testing text
	@for i in $(shell seq 1 10) ; do \
		echo "Running $$i.txt"; \
		./mydisambig segmented_test_data/$$i.txt res/ZhuYin-Big5.map bigram  result2/$$i.txt; \
	done;

exp:
	@#TODO experiment for how fast and the result of disambig
	@for i in $(shell seq 1 10) ; do \
		echo "Running $$i.txt"; \
		disambig -text segmented_test_data/$$i.txt -map res/ZhuYin-Big5.map -lm bigram -order 2 > result2-1/$$i.txt;\
	done;

map:
	@#TODO
	@python3 $(SRC_PATH)/mapping.py $(FROM) $(TO)

clean:
	$(RM) $(OBJ) $(TARGET)
