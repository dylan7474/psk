# Makefile for C/SDL2 projects compiled for Linux
#
# This Makefile uses sdl2-config to automatically find the correct
# compiler and linker flags, making it portable across Linux systems.

# The C compiler to use.
CC = gcc

# The name of the final Linux executable.
TARGET = psk

# All C source files used in the project.
SRCS = main.c

# Use sdl2-config to get the compiler flags for SDL2.
CFLAGS = -Wall -O2 $(shell sdl2-config --cflags)

# Use sdl2-config for the base SDL2 library, and add others manually.
LDFLAGS = $(shell sdl2-config --libs) -lSDL2_mixer -lm -lSDL2_ttf

# --- Build Rules ---

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(SRCS) -o $(TARGET) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

