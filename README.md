# enn: A Neural Network from Scratch, written in C
Explainable Neural Network written in C, designed for teaching how neural networks work in depth.

## Build

To build, run `make`. To enable debug flags, run `make EFLAGS=-g`.

## Roadmap

### Done
- Linalg, NN, data structs, main files
- Linalg functions (mul, add, scale, transpose)
- Some tests
- Seperate into src, test, build folders
- Documentation (Javadoc style) comments
- Convert loss functions from Matrix* to double and use function pointers

### Not Done
- Write errors in a functional way: https://softwareengineering.stackexchange.com/questions/420872/how-functional-programming-achieves-no-runtime-exceptions 
- Add mnewo/mcreat/mnew2 function to allocate only if NULL or rows and cols incorrect.
- Add optional out argument to linalg functions for mutability.
- Add NULL checks to linalg functions
- Test newly created functions (losses)
- Add backprop
- CMake
- Valgrind
- Tutorial book
- Normalizer
- Functional prog headers
- Working Feedforward NN
- Write in functional style
- Read from CSV
- Command line options
- Save weight functionality
- Null checks
- run infer (`/opt/infer-linux64-v0.17.0/bin/infer -- make`)
