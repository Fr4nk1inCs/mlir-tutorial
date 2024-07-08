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

using TensorShape = std::vector<int64_t>;

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

  ExpressionKind getKind() const { return kind; }

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

  double getValue() { return value; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Number;
  }
};

/// Tensor literal like `[[1.0 2.0] [3.0 4.0]]`
class TensorLiteral : public Expression {
  std::vector<std::unique_ptr<Expression>> values;
  TensorShape shape;

public:
  TensorLiteral(Location location,
                std::vector<std::unique_ptr<Expression>> values,
                TensorShape shape)
      : Expression(ExpressionKind::Tensor, std::move(location)),
        values(std::move(values)), shape(std::move(shape)) {}

  llvm::ArrayRef<std::unique_ptr<Expression>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getShape() { return shape; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Tensor;
  }
};

/// Variable like `a`
class Variable : public Expression {
  std::string name;

public:
  Variable(Location location, std::string name)
      : Expression(ExpressionKind::Variable, std::move(location)),
        name(std::move(name)) {}

  llvm::StringRef getName() { return name; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Variable;
  }
};

/// Declarition of variable like `a = [1 2 3]`
class VariableDecl : public Expression {
  std::string name;
  TensorShape shape;
  std::unique_ptr<Expression> initValue;

public:
  VariableDecl(Location location, std::string name, TensorShape shape,
               std::unique_ptr<Expression> initValue)
      : Expression(ExpressionKind::VariableDecl, std::move(location)),
        name(std::move(name)), shape(std::move(shape)),
        initValue(std::move(initValue)) {}

  llvm::StringRef getName() { return name; }
  Expression *getInitValue() { return initValue.get(); }
  const TensorShape &getShape() { return shape; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::VariableDecl;
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

  std::optional<Expression *> getValue() {
    if (value.has_value())
      return value->get();
    return std::nullopt;
  }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Return;
  }
};

/// Print expression like `print(a)`
class PrintExpr : public Expression {
  std::unique_ptr<Expression> arg;

public:
  PrintExpr(Location location, std::unique_ptr<Expression> arg)
      : Expression(ExpressionKind::Print, std::move(location)),
        arg(std::move(arg)) {}

  Expression *getArg() { return arg.get(); }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Print;
  }
};

/// Transpose expression like `transpose(a)`
class TransposeExpr : public Expression {
  std::unique_ptr<Expression> arg;

public:
  TransposeExpr(Location location, std::unique_ptr<Expression> arg)
      : Expression(ExpressionKind::Transpose, std::move(location)),
        arg(std::move(arg)) {}

  Expression *getArg() { return arg.get(); }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Transpose;
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

  char getOp() { return op; }
  Expression *getLhs() { return lhs.get(); }
  Expression *getRhs() { return rhs.get(); }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Binary;
  }
};

/// Calling function like `func(a, b)`
class CallExpr : public Expression {
  std::string callee;
  std::vector<std::unique_ptr<Expression>> args;

public:
  CallExpr(Location location, std::string function,
           std::vector<std::unique_ptr<Expression>> args)
      : Expression(ExpressionKind::Call, std::move(location)),
        callee(std::move(function)), args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<Expression>> getArgs() { return args; }

  /// LLVM-style RTTI
  static bool classof(const Expression *e) {
    return e->getKind() == ExpressionKind::Call;
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
      : location(std::move(location)), name(std::move(name)),
        params(std::move(params)) {}

  llvm::StringRef getName() { return name; }
  llvm::ArrayRef<std::unique_ptr<Variable>> getParams() { return params; }
  const Location &loc() { return location; }
};

class Function {
  std::unique_ptr<Prototype> prototype;
  std::unique_ptr<ExpressionList> body;

public:
  Function(const Function &) = delete;
  Function(Function &&) = default;
  Function &operator=(const Function &) = delete;
  Function &operator=(Function &&) = default;
  Function(std::unique_ptr<Prototype> prototype,
           std::unique_ptr<ExpressionList> body)
      : prototype(std::move(prototype)), body(std::move(body)) {}

  Prototype *getPrototype() { return prototype.get(); }
  ExpressionList *getBody() { return body.get(); }
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
