# To run infer: /opt/infer-linux64-v0.17.0/bin/infer run -- make
CC=gcc
CFLAGS=-std=c90 -pedantic -Wall -Wextra $(EFLAGS)

SRC_DIR=./src
BIN_DIR=./build
TEST_DIR=./test

SRC=$(wildcard $(SRC_DIR)/*.c)
OBJECTS=$(patsubst %.c, %.o, $(SRC))

TEST_SRC=$(wildcard $(TEST_DIR)/*.c)
TEST_OBJS=$(patsubst %.c, %.o, $(TEST_SRC))

EXECUTABLE=$(BIN_DIR)/enn
TEST_EXE=$(BIN_DIR)/enn_test

all: $(SOURCES) $(EXECUTABLE) $(TEST_SRC) $(TEST_EXE)

clean:
	rm -f $(SRC_DIR)/*.o
	rm -f $(BIN_DIR)/enn
	rm -f $(TEST_DIR)/*.o

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

$(TEST_EXE): $(TEST_OBJS)
	$(CC) $(CFLAGS) $(filter-out $(SRC_DIR)/main.o, $(OBJECTS)) $(TEST_OBJS) -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
