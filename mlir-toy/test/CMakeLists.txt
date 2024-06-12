file(GLOB TEST_FILES *.toy)

foreach(filepath ${TEST_FILES})
  cmake_path(GET filepath STEM LAST_ONLY file)
  add_test(
    NAME ${file}
    COMMAND
    bash -c "${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh ${CMAKE_BINARY_DIR}/mlir-toy ${file}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endforeach()
