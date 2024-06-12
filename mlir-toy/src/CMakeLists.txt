add_subdirectory(parser)

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
