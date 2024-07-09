#pragma once

#ifndef TOY_OPS_H
#define TOY_OPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Toy/ToyDialect.h"

#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif
