#ifndef TOY_AST_HPP
#define TOY_AST_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace toy {
namespace AST {

struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

using TensorShape = std::vector<uint64_t>;

/// Base class for a expression
class Expression {
public:
  enum ExpressionKind {
    Number,
    Tensor,
    Variable,
    VariableDecl,
    Return,
    Print,
    Transpose,
    Binary,
    Call,
  };

  Expression(ExpressionKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~Expression() = default;

  ExpressionKind get_kind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExpressionKind kind;
  Location location;
};

/// Expression list in a block
using ExpressionList = std::vector<std::unique_ptr<Expression>>;

/// Number literal like `1.0`
class NumberLiteral : public Expression {
  double value;

public:
  NumberLiteral(Location location, double value)
      : Expression(ExpressionKind::Number, std::move(location)), value(value) {}

  double get_value() { return value; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Number;
  }
};

/// Tensor literal like `[[1.0 2.0] [3.0 4.0]]`
class TensorLiteral : public Expression {
  std::vector<std::unique_ptr<Expression>> values;
  std::vector<uint64_t> dimensions;

public:
  TensorLiteral(Location location,
                std::vector<std::unique_ptr<Expression>> values,
                std::vector<uint64_t> dimensions)
      : Expression(ExpressionKind::Tensor, std::move(location)),
        values(std::move(values)), dimensions(dimensions) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> get_values() { return values; }
  llvm::ArrayRef<uint64_t> get_dims() { return dimensions; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Tensor;
  }
};

/// Variable like `a`
class Variable : public Expression {
  std::string name;

public:
  Variable(Location location, std::string name)
      : Expression(ExpressionKind::Variable, std::move(location)), name(name) {}

  llvm::StringRef get_name() { return name; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Variable;
  }
};

/// Declarition of variable like `a = [1 2 3]`
class VariableDecl : public Expression {
  std::string name;
  TensorShape shape;
  std::unique_ptr<Expression> init_value;

public:
  VariableDecl(Location location, std::string name, TensorShape shape,
               std::unique_ptr<Expression> init_value)
      : Expression(ExpressionKind::VariableDecl, std::move(location)),
        name(name), shape(std::move(shape)), init_value(std::move(init_value)) {
  }

  llvm::StringRef get_name() { return name; }
  Expression *get_init_value() { return init_value.get(); }
  const TensorShape &get_shape() { return shape; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::VariableDecl;
  }
};

/// Return expression: `return` or `return a`
class ReturnExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> value;

public:
  ReturnExpr(Location location,
             std::optional<std::unique_ptr<Expression>> value)
      : Expression(ExpressionKind::Return, std::move(location)),
        value(std::move(value)) {}

  std::optional<Expression *> get_value() {
    if (value.has_value())
      return value->get();
    return std::nullopt;
  }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Return;
  }
};

/// Print expression like `print(a)`
class PrintExpr : public Expression {
  std::unique_ptr<Expression> arg;

public:
  PrintExpr(Location location, std::unique_ptr<Expression> arg)
      : Expression(ExpressionKind::Print, std::move(location)),
        arg(std::move(arg)) {}

  Expression *get_arg() { return arg.get(); }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Print;
  }
};

/// Transpose expression like `transpose(a)`
class TransposeExpr : public Expression {
  std::unique_ptr<Expression> arg;

public:
  TransposeExpr(Location location, std::unique_ptr<Expression> arg)
      : Expression(ExpressionKind::Transpose, std::move(location)),
        arg(std::move(arg)) {}

  Expression *get_arg() { return arg.get(); }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Transpose;
  }
};

/// Binary operations like a * b
class BinaryExpr : public Expression {
  char op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;

public:
  BinaryExpr(Location location, char op, std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
      : Expression(ExpressionKind::Binary, std::move(location)), op(op),
        lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  char get_op() { return op; }
  Expression *get_lhs() { return lhs.get(); }
  Expression *get_rhs() { return rhs.get(); }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Binary;
  }
};

/// Calling function like `func(a, b)`
class CallExpr : public Expression {
  std::string callee;
  std::vector<std::unique_ptr<Expression>> args;

public:
  CallExpr(Location location, std::string function,
           std::vector<std::unique_ptr<Expression>> args)
      : Expression(ExpressionKind::Call, std::move(location)), callee(function),
        args(std::move(args)) {}

  llvm::StringRef get_callee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<Expression>> get_args() { return args; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->get_kind() == ExpressionKind::Call;
  }
};

/// Function prototype like func(arg1, arg2)
/// Contains the name and parameters (as variable) of a function
class Prototype {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<Variable>> params;

public:
  Prototype(Location location, std::string name,
            std::vector<std::unique_ptr<Variable>> params)
      : location(std::move(location)), name(name), params(std::move(params)) {}

  llvm::StringRef get_name() { return name; }
  llvm::ArrayRef<std::unique_ptr<Variable>> get_params() { return params; }
  const Location &loc() { return location; }
};

class Function {
  std::unique_ptr<Prototype> prototype;
  std::unique_ptr<ExpressionList> body;

public:
  Function(std::unique_ptr<Prototype> prototype,
           std::unique_ptr<ExpressionList> body)
      : prototype(std::move(prototype)), body(std::move(body)) {}

  Prototype *get_prototype() { return prototype.get(); }
  ExpressionList *get_body() { return body.get(); }
};

class Module {
  std::vector<Function> functions;

public:
  Module(std::vector<Function> functions) : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }
};

void dump(Module &module);
} // namespace AST
} // namespace toy

#endif // TOY_AST_HPP
