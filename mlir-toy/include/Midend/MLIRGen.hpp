#pragma once

#ifndef TOY_MLIRGEN_HPP
#define TOY_MLIRGEN_HPP

#include "Toy/AST.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace toy {

/// Emit MLIR for the given Toy AST::module, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          AST::Module &module);
} // namespace toy

#endif // TOY_MLIRGEN_HPP
