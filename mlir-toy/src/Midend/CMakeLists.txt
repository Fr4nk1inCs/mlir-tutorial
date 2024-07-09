add_library(ToyMLIRGenerator
  MLIRGen.cpp
)

target_link_libraries(ToyMLIRGenerator
  ToyDialect
)
