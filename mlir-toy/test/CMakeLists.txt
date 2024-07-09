file(GLOB TOY_FILES *.toy)

foreach(filepath ${TOY_FILES})
  cmake_path(GET filepath STEM LAST_ONLY file)
  add_test(
    NAME ${file}-toy
    COMMAND
    bash -c "${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh ${CMAKE_BINARY_DIR}/mlir-toy ${file}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endforeach()

file(GLOB MLIR_FILES *.mlir)

foreach(filepath ${MLIR_FILES})
  cmake_path(GET filepath STEM LAST_ONLY file)
  add_test(
    NAME ${file}-mlir
    COMMAND
    bash -c "${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh ${CMAKE_BINARY_DIR}/mlir-toy ${file}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endforeach()
