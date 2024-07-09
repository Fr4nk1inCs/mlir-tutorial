//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "Midend/MLIRGen.hpp"
#include "Toy/AST.hpp"
#include "Toy/ToyDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(AST::Module &module) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (AST::Function &f : module)
      mlirGen(f);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(const AST::Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::toy::FuncOp mlirGen(AST::Prototype &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getParams().size(),
                                              getType(AST::TensorShape{}));
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<mlir::toy::FuncOp>(location, proto.getName(),
                                             funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::toy::FuncOp mlirGen(AST::Function &func) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::toy::FuncOp function = mlirGen(*func.getPrototype());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = func.getPrototype()->getParams();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*func.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    mlir::toy::ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<mlir::toy::ReturnOp>(loc(func.getPrototype()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(AST::TensorShape{})));
    }

    return function;
  }

  /// Emit a binary operation
  mlir::Value mlirGen(AST::BinaryExpr &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLhs());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRhs());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<mlir::toy::AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<mlir::toy::MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(AST::Variable &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(AST::ReturnExpr &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getValue().has_value()) {
      if (!(expr = mlirGen(**ret.getValue())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<mlir::toy::ReturnOp>(
        location, expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen(AST::TensorLiteral &lit) {
    auto type = getType(lit.getShape());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getShape().begin(), lit.getShape().end(),
                                 1, std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getShape(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<mlir::toy::ConstantOp>(loc(lit.loc()), type,
                                                 dataAttribute);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(AST::Expression &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<AST::TensorLiteral>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<AST::NumberLiteral>(expr) && "expected literal or number expr");
    data.push_back(cast<AST::NumberLiteral>(expr).getValue());
  }

  /// Emit a call expression. Identifiers are assumed to be user-defined
  /// functions.
  mlir::Value mlirGen(AST::CallExpr &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Calls to user-defined functions are mapped to a custom call that takes
    // the callee name as an attribute.
    return builder.create<mlir::toy::GenericCallOp>(location, callee, operands);
  }

  /// Emit a print expression. It emits specific builtin operations print(x).
  mlir::LogicalResult mlirGen(AST::PrintExpr &call) {
    auto arg = mlirGen(*call.getArg());
    auto location = loc(call.loc());

    if (!arg) {
      emitError(location, "MLIR codegen encountered an error: toy.print "
                          "does not accept multiple arguments");
      return mlir::failure();
    }

    builder.create<mlir::toy::PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a transpose expression. It emits specific builtin operations
  /// transpose(x).
  mlir::LogicalResult mlirGen(AST::TransposeExpr &call) {
    auto arg = mlirGen(*call.getArg());
    auto location = loc(call.loc());

    if (!arg) {
      emitError(location, "MLIR codegen encountered an error: toy.transpose "
                          "does not accept multiple arguments");
      return mlir::failure();
    }

    builder.create<mlir::toy::TransposeOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(AST::NumberLiteral &num) {
    return builder.create<mlir::toy::ConstantOp>(loc(num.loc()),
                                                 num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(AST::Expression &expr) {
    switch (expr.getKind()) {
    case toy::AST::Expression::Binary:
      return mlirGen(cast<AST::BinaryExpr>(expr));
    case toy::AST::Expression::Variable:
      return mlirGen(cast<AST::Variable>(expr));
    case toy::AST::Expression::Tensor:
      return mlirGen(cast<AST::TensorLiteral>(expr));
    case toy::AST::Expression::Call:
      return mlirGen(cast<AST::CallExpr>(expr));
    case toy::AST::Expression::Number:
      return mlirGen(cast<AST::NumberLiteral>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(AST::VariableDecl &vardecl) {
    auto *init = vardecl.getInitValue();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getShape().empty()) {
      value = builder.create<mlir::toy::ReshapeOp>(
          loc(vardecl.loc()), getType(vardecl.getShape()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(AST::ExpressionList &block) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
    for (auto &expr : block) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<AST::VariableDecl>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<AST::ReturnExpr>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<AST::PrintExpr>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }
      if (auto *transpose = dyn_cast<AST::TransposeExpr>(expr.get())) {
        if (mlir::failed(mlirGen(*transpose)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }
};

} // namespace

namespace toy {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          AST::Module &module) {
  return MLIRGenImpl(context).mlirGen(module);
}

} // namespace toy
