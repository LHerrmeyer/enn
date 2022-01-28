CC=gcc
CFLAGS=-std=c90 -pedantic -Wall -Wextra
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
