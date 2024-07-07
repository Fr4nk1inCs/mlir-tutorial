add_subdirectory(parser)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(mlir-toy main.cpp)

target_link_libraries(
  mlir-toy
  parser
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)

install(
  TARGETS mlir-toy
  RUNTIME DESTINATION bin
)
