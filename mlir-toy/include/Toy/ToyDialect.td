#ifndef TOY_DIALECT_TD
#define TOY_DIALECT_TD

include "mlir/IR/OpBase.td"

// Provide a definition of the 'toy' dialect in the ODS framework.
def ToyDialect : Dialect {
  let name = "toy";
  let cppNamespace = "mlir::toy";
  let summary = "Toy Dialect";
  let description = [{
    Toy dialect for the MLIR system.
  }];
}

#endif
