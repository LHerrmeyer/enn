# To run infer: /opt/infer-linux64-v0.17.0/bin/infer run -- make
# For clang static analyzer (package clang-tools): scan-build make
# valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./build/enn_test
CC=gcc
CFLAGS=-std=c90 -pedantic -Wall -Wextra $(EFLAGS)
LDFLAGS=-lm

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
	rm -f $(BIN_DIR)/enn_test
	rm -f $(TEST_DIR)/*.o

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

$(TEST_EXE): $(TEST_OBJS)
	$(CC) $(CFLAGS) $(filter-out $(SRC_DIR)/main.o, $(OBJECTS)) $(TEST_OBJS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@
