CC = gcc
CFLAGS = -O3 -Wall
TARGET = API_negamax
SRCS = main.c board.c move.c search.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)