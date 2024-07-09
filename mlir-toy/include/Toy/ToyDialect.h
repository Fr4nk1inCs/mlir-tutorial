#pragma once

#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "Toy/ToyOpsDialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif
