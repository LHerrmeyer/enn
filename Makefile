# To run infer: /opt/infer-linux64-v0.17.0/bin/infer run -- make
CC=gcc
CFLAGS=-std=c90 -pedantic -Wall -Wextra $(EFLAGS)
SOURCES=linalg.c main.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=enn

all: $(SOURCES) $(EXECUTABLE)

clean:
	rm *.o

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
