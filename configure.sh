#!/bin/bash
#
# A generic configure script to check for build dependencies for C/SDL2 projects.
# It checks for the necessary tools and libraries for both Linux builds
# and optional Windows cross-compilation.
#

echo "--- Checking Build Environment ---"

# --- Configuration ---
# Color codes for output
NC='\033[0m' # No Color
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'

# Status trackers
SUCCESS=0
FAILURE=1
ALL_OK=true
MISSING_DEPS=""

# --- Helper Functions ---

# check_command: Checks if a command exists on the system.
# Arguments:
#   $1: The command to check for (e.g., "gcc").
#   $2: The package name that provides this command (e.g., "build-essential").
check_command() {
    local cmd=$1
    local pkg=$2
    echo -n "Checking for command '$cmd'... "
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}found${NC}"
        return $SUCCESS
    else
        echo -e "${RED}not found${NC}"
        ALL_OK=false
        # Add the corresponding package to the list of missing dependencies.
        # This avoids adding the same package multiple times.
        if [[ $MISSING_DEPS != *"$pkg"* ]]; then
            MISSING_DEPS+="$pkg "
        fi
        return $FAILURE
    fi
}

# --- Main Dependency Checks ---

echo ""
echo "--- Checking Linux Build Tools ---"
check_command "gcc" "build-essential"
check_command "make" "build-essential"
check_command "xxd" "xxd"

# The 'sdl2-config' script is the most reliable way to check for the core
# SDL2 development library and get its compiler flags.
if check_command "sdl2-config" "libsdl2-dev"; then
    # If sdl2-config exists, we can perform more specific checks for the other libraries.
    # We use the C pre-processor (-E) to see if the header files can be found.
    SDL_CFLAGS=$(sdl2-config --cflags)

    echo -n "Checking for library 'SDL2_mixer'... "
    if echo "#include <SDL2/SDL_mixer.h>" | gcc -E $SDL_CFLAGS - &>/dev/null; then
        echo -e "${GREEN}found${NC}"
    else
        echo -e "${RED}not found${NC}"
        ALL_OK=false
        MISSING_DEPS+="libsdl2-mixer-dev "
    fi

    echo -n "Checking for library 'SDL2_ttf'... "
    if echo "#include <SDL2/SDL_ttf.h>" | gcc -E $SDL_CFLAGS - &>/dev/null; then
        echo -e "${GREEN}found${NC}"
    else
        echo -e "${RED}not found${NC}"
        ALL_OK=false
        MISSING_DEPS+="libsdl2-ttf-dev "
    fi

    echo -n "Checking for library 'SDL2_image'... "
    if echo "#include <SDL2/SDL_image.h>" | gcc -E $SDL_CFLAGS - &>/dev/null; then
        echo -e "${GREEN}found${NC}"
    else
        echo -e "${RED}not found${NC}"
        ALL_OK=false
        MISSING_DEPS+="libsdl2-image-dev "
    fi
else
    # If sdl2-config is missing, we can't check for the others and should add them all.
    echo -e "${YELLOW}Skipping dependent SDL2 library checks.${NC}"
    MISSING_DEPS+="libsdl2-mixer-dev libsdl2-ttf-dev libsdl2-image-dev "
fi


echo ""
echo "--- Checking Windows Cross-Compilation Tools (Optional) ---"
if check_command "x86_64-w64-mingw32-gcc" "mingw-w64"; then
    # You could add more detailed checks here for the MinGW versions of the SDL libs if needed.
    : # This is a no-op, just to have a valid 'then' block.
fi


# --- Final Summary ---
echo "----------------------------------------------------"
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}Environment is correctly configured!${NC}"
    echo "You can now run 'make' to build for Linux."
    echo "To build for Windows, run 'make -f Makefile.win'."
    exit 0
else
    echo -e "${RED}Configuration failed. Missing dependencies.${NC}"
    echo "Please install the following required packages."
    echo "On a Debian/Ubuntu based system, you can use this command:"
    
    # Create a unique, sorted list of packages to install.
    UNIQUE_DEPS=$(echo "$MISSING_DEPS" | tr ' ' '\n' | sort -u | tr '\n' ' ')
    echo -e "${YELLOW}sudo apt-get install $UNIQUE_DEPS${NC}"
    
    exit 1
fi
