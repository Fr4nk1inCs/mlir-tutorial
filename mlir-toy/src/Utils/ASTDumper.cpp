#include "Toy/AST.hpp"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy::AST;

namespace {

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { level--; }

private:
  int &level;
};

template <class T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

} // namespace

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class Dumper {
public:
  void dump(Module *module);

private:
  void dump(const TensorShape &shape);
  void dump(Function *func);
  void dump(Prototype *proto);
  void dump(ExpressionList *exprs);
  void dump(Expression *expr);
  void dump(PrintExpr *expr);
  void dump(CallExpr *expr);
  void dump(BinaryExpr *expr);
  void dump(TransposeExpr *expr);
  void dump(ReturnExpr *expr);
  void dump(VariableDecl *decl);
  void dump(Variable *var);
  void dump(NumberLiteral *num);
  void dump(TensorLiteral *tensor);

  void printIndent() const {
    for (int i = 0; i < level; ++i)
      llvm::errs() << "  ";
  }

  int level = 0;
};

#define INDENT()                                                               \
  Indent level_(level);                                                        \
  printIndent();

void Dumper::dump(const TensorShape &shape) {
  llvm::errs() << "<";
  llvm::interleaveComma(shape, llvm::errs());
  llvm::errs() << ">";
}

void Dumper::dump(Expression *expr) {
  llvm::TypeSwitch<Expression *>(expr)
      .Case<BinaryExpr, CallExpr, PrintExpr, TransposeExpr, ReturnExpr,
            NumberLiteral, TensorLiteral, Variable, VariableDecl>(
          [&](auto *node) { this->dump(node); })
      .Default([&](Expression *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}

void Dumper::dump(PrintExpr *expr) {
  INDENT();
  llvm::errs() << "Print [ " << loc(expr) << "\n";
  dump(expr->getArg());
  printIndent();
  llvm::errs() << "]\n";
}

void Dumper::dump(CallExpr *expr) {
  INDENT();
  llvm::errs() << "Call " << expr->getCallee() << "' [ " << loc(expr) << "\n";
  for (auto &arg : expr->getArgs())
    dump(arg.get());
  printIndent();
  llvm::errs() << "]\n";
}

void Dumper::dump(BinaryExpr *expr) {
  INDENT();
  llvm::errs() << "BinOp: " << expr->getOp() << " " << loc(expr) << "\n";
  dump(expr->getLhs());
  dump(expr->getRhs());
}

void Dumper::dump(TransposeExpr *expr) {
  INDENT();
  llvm::errs() << "Call 'transpose' [ " << loc(expr) << "\n";
  dump(expr->getArg());
  printIndent();
  llvm::errs() << "]\n";
}

void Dumper::dump(ReturnExpr *expr) {
  INDENT();
  llvm::errs() << "Return " << loc(expr) << "\n";
  if (expr->getValue().has_value())
    dump(expr->getValue().value());
  else {
    INDENT();
    llvm::errs() << "(void)\n";
  }
}

void Dumper::dump(VariableDecl *decl) {
  INDENT();
  llvm::errs() << "VarDecl " << decl->getName();
  dump(decl->getShape());
  llvm::errs() << " " << loc(decl) << "\n";
  dump(decl->getInitValue());
}

void Dumper::dump(Variable *var) {
  INDENT();
  llvm::errs() << "var: " << var->getName() << " " << loc(var) << "\n";
}

void Dumper::dump(NumberLiteral *num) {
  INDENT();
  llvm::errs() << "number: " << num->getValue() << " " << loc(num) << "\n";
}

/// Helper function to print recursively a tensor.
/// For tensor like: [ [ 1, 2 ], [ 3, 4 ] ]
/// It will print the tensor along with its shape:
///     <2,2>[ <2>[ 1, 2 ], <2>[ 3, 4 ] ]
void dumpTensorHelper(Expression *tensorOrNumber) {
  // Number
  if (auto *number = llvm::dyn_cast<NumberLiteral>(tensorOrNumber)) {
    llvm::errs() << number->getValue();
    return;
  }
  // Tensor
  auto *tensor = llvm::cast<TensorLiteral>(tensorOrNumber);

  if (tensor == nullptr) {
    llvm::errs() << "<null>";
    return;
  }

  // Print shape first
  llvm::errs() << "<";
  llvm::interleaveComma(tensor->getShape(), llvm::errs());
  llvm::errs() << ">";

  // Print values
  llvm::errs() << "[ ";
  llvm::interleaveComma(tensor->getValues(), llvm::errs(),
                        [&](auto &expr) { dumpTensorHelper(expr.get()); });
  llvm::errs() << " ]";
}

void Dumper::dump(TensorLiteral *tensor) {
  INDENT();
  llvm::errs() << "Literal: ";
  dumpTensorHelper(tensor);
  llvm::errs() << " " << loc(tensor) << "\n";
}

void Dumper::dump(ExpressionList *exprs) {
  INDENT();
  llvm::errs() << "Block {\n";
  for (auto &expr : *exprs)
    dump(expr.get());
  printIndent();
  llvm::errs() << "} // Block\n";
}

void Dumper::dump(Prototype *proto) {
  INDENT();
  llvm::errs() << "Proto '" << proto->getName() << "' " << loc(proto) << "\n";
  printIndent();
  llvm::errs() << "Params: [";
  llvm::interleaveComma(proto->getParams(), llvm::errs(),
                        [](auto &param) { llvm::errs() << param->getName(); });
  llvm::errs() << "]\n";
}

void Dumper::dump(Function *func) {
  INDENT();
  llvm::errs() << "Function \n";
  dump(func->getPrototype());
  dump(func->getBody());
}

void Dumper::dump(Module *module) {
  INDENT();
  llvm::errs() << "Module: \n";
  for (auto &func : *module)
    dump(&func);
}

void toy::AST::dump(Module &module) { Dumper().dump(&module); };
