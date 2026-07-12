# MLIR Toy Tutorial

A hands-on tutorial project for learning MLIR (Multi-Level Intermediate Representation) through the Toy language example from the official MLIR documentation.

## Overview

This project implements a simple custom language called "Toy" to demonstrate MLIR concepts including:
- Dialect definition
- Operation (Op) definition using ODS (Operation Definition Specification)
- MLIR pass development
- Lowering from high-level to low-level representations
- Using MLIR's C++ API

This tutorial follows the [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) structure but with practical, buildable code.

## Project Structure

```
toy-tutorial/
├── CMakeLists.txt          # Build configuration
├── Tutorial.md             # Detailed tutorial documentation
├── ods/                    # ODS files for operation definitions
├── src/                    # C++ source code
│   ├── AST.h              # Abstract Syntax Tree definitions
│   ├── Lexer.h/cpp        # Lexer implementation
│   ├── Parser.h/cpp       # Parser implementation
│   ├── MLIRGen.h/cpp      # MLIR generation from AST
│   ├── Dialect.cpp        # Toy dialect definition
│   ├── Ops.td             # Operation definitions
│   └── Transforms/        # MLIR pass implementations
├── input/                  # Example Toy language programs
└── build/                 # Build directory (generated)
```

## Prerequisites

- C++ compiler with C++17 support
- CMake 3.13+
- MLIR built from LLVM (follow setup in parent project's CLAUDE.md)

### Environment Setup

If you haven't built MLIR yet, refer to the main project's [CLAUDE.md](../../CLAUDE.md) for instructions, or use:

```bash
# Assuming LLVM is cloned to ../llvm-project
cmake -G Ninja -S ../llvm-project/llvm -B ../llvm-project/build \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release

cmake --build ../llvm-project/build --target mlir-opt mlir-translate
```

## Building

```bash
cd build
cmake -G Ninja \
  -DMLIR_DIR=$PWD/../../llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../../llvm-project/build/lib/cmake/llvm \
  ..
ninja
```

## Usage

After building, you can:

1. **Run the Toy compiler:**
   ```bash
   ./build/toyc input/example.toy
   ```

2. **Test MLIR passes:**
   ```bash
   ./build/toyc input/example.toy -emit=mlir
   ```

3. **View the tutorial documentation:**
   See [Tutorial.md](Tutorial.md) for step-by-step explanations.

## Learning Path

1. Understand the Toy language syntax (see `input/` examples)
2. Follow the lexer and parser implementation
3. Learn how AST is lowered to MLIR
4. Study the custom dialect and operation definitions
5. Explore MLIR passes for optimization and lowering

## References

- [MLIR Official Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [ODS Reference](https://mlir.llvm.org/docs/OpDefinitions/)

## License

This project is part of the [tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn) learning repository, intended for educational purposes.

---

**Part of:** [tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn) - A comprehensive learning path for deep learning compilers.
