cmake_minimum_required(VERSION 3.20)
project(ToyTutorial LANGUAGES C CXX)

# Configure with the directory that contains MLIRConfig.cmake, for example:
# cmake -S . -B build -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
# Some distro packages install MLIR's CMake files and mlir-tblgen below the
# same prefix, while LLVM's exported target can still point at a missing system
# location. Prefer the tool shipped next to the selected MLIRConfig.cmake.
get_filename_component(MLIR_CONFIG_TOOLS_DIR "${MLIR_DIR}/../../../bin" ABSOLUTE)
if(EXISTS "${MLIR_CONFIG_TOOLS_DIR}/mlir-tblgen")
  set(MLIR_TABLEGEN_EXE "${MLIR_CONFIG_TOOLS_DIR}/mlir-tblgen")
endif()
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)

# ODS/TableGen example. These generated files are intentionally kept in the
# build directory: the .td file remains the single source of truth.
# `mlir_tablegen` reads this directory property when preparing its -I flags.
include_directories(${MLIR_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS ods/OdsToyCostOpInterface.td)
mlir_tablegen(OdsToyCostOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(OdsToyCostOpInterface.cpp.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS ods/OdsToyOps.td)
mlir_tablegen(OdsToyOps.h.inc -gen-op-decls)
mlir_tablegen(OdsToyOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(OdsToyIncGen)

add_executable(ods-toy-opt src/04_ods-toy-opt.cpp)
add_dependencies(ods-toy-opt OdsToyIncGen)
target_compile_features(ods-toy-opt PRIVATE cxx_std_17)
target_include_directories(ods-toy-opt PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(ods-toy-opt PRIVATE
  MLIRMlirOptMain
  MLIRArithmetic
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
)

add_executable(toy-opt src/01_toy-opt.cpp)
target_compile_features(toy-opt PRIVATE cxx_std_17)
target_include_directories(toy-opt PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
target_link_libraries(toy-opt PRIVATE
  MLIROptLib
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
)

add_executable(pass-examples src/02_pass-examples.cpp)
target_compile_features(pass-examples PRIVATE cxx_std_17)
target_include_directories(pass-examples PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(pass-examples PRIVATE
  MLIROptLib
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
)

# A deliberately small driver for studying pass nesting and greedy rewrites.
add_executable(pass-manager-greedy src/05_pass-manager-greedy.cpp)
target_compile_features(pass-manager-greedy PRIVATE cxx_std_17)
target_include_directories(pass-manager-greedy PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(pass-manager-greedy PRIVATE
  MLIROptLib
  MLIRArithmetic
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
)

add_executable(lowering-examples src/03_lowering--examples.cpp)
target_compile_features(lowering-examples PRIVATE cxx_std_17)
target_include_directories(lowering-examples PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(lowering-examples PRIVATE
  MLIROptLib
  MLIRAffine
  MLIRAffineToStandard
  MLIRArithmetic
  MLIRStandard
  MLIRLLVMIR
  MLIRMemRef
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCF
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
)
