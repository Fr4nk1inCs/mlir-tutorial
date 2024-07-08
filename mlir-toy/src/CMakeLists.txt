add_subdirectory(Utils)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(mlir-toy main.cpp)

target_link_libraries(
  mlir-toy
  ToyASTDumper
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)

install(
  TARGETS mlir-toy
  RUNTIME DESTINATION bin
)
