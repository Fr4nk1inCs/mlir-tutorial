# MLIR Toy Dialect

## Build

1. Build LLVM & MLIR into `install/`
   ```bash
   # configure
   mkdir install
   cd llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=ON \
       -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DLLVM_ENABLE_LLD=ON \
       -DLLVM_CCACHE_BUILD=ON \
       -DLLVM_INSTALL_UTILS=ON \
       -DLLVM_ENABLE_RTTI=ON \
       -DCMAKE_INSTALL_PREFIX="../../install"
   # build & install
   cmake --build . && cmake --install .
   ```
2. Build `toy` dialect (out-of-tree)
   ```bash
   # configure
   cd mlir-toy
   mkdir build && cd build
   cmake -GNinja .. -DCMAKE_INSTALL_PREFIX=../../install
   # build
   cmake --build .
   ```
