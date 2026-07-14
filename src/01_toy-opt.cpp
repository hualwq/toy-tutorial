// A deliberately small, handwritten version of the first Toy tutorial ideas.
// Production dialects normally use ODS/TableGen; see mlir-ods/ afterwards.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace toy {

// An operation class gives a textual operation a C++ API and declares its
// structural contract. The generic MLIR assembly form supplies its operands,
// result type, and attributes, so no custom parser is needed in this example.
class AddOp : public Op<AddOp, OpTrait::NOperands<2>::Impl,
                        OpTrait::OneResult, OpTrait::ZeroRegions> { // Region: 理解成TVM中的Block，OpTrait::ZeroRegions表示没有Region
public:
  using Op::Op;
  static StringRef getOperationName() { return "toy.add"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }

  LogicalResult verify() {
    if (!getLhs().getType().isF32() || !getRhs().getType().isF32() ||
        !getResult().getType().isF32())
      return emitOpError("requires two f32 operands and one f32 result");
    return success();
  }
};

class MultiplyOp : public Op<MultiplyOp, OpTrait::NOperands<2>::Impl,
                             OpTrait::OneResult, OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "toy.multiply"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }

  LogicalResult verify() {
    if (!getLhs().getType().isF32() || !getRhs().getType().isF32() ||
        !getResult().getType().isF32())
      return emitOpError("requires two f32 operands and one f32 result");
    return success();
  }
};

// A dialect is a namespace plus the operations/types/attributes it owns.
class ToyDialect : public Dialect {
public:
  explicit ToyDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<ToyDialect>()) {
    addOperations<AddOp, MultiplyOp>();
  }

  static StringRef getDialectNamespace() { return "toy"; }
};

// Each pattern states one local semantic-preserving rewrite. Notice that the
// target operation belongs to a different dialect: this is a dialect lowering.
class LowerAdd : public OpRewritePattern<AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

class LowerMultiply : public OpRewritePattern<MultiplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiplyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};

struct LowerToyToArithPass
    : public PassWrapper<LowerToyToArithPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToyToArithPass)

  StringRef getArgument() const final { return "lower-toy-to-arith"; }
  StringRef getDescription() const final {
    return "Replace toy scalar arithmetic with arith operations";
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());
    // 指定转换的目标，哪些Dialect是合法的，哪些Dialect是非法的
    target.addIllegalDialect<ToyDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();

    //重写规则集合，加入Add，Multiply的重写规则
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAdd, LowerMultiply>(&getContext());
    // 将三样东西交给MLIR的 dialect conversation：
    // 1. getOperation() 返回的是继承参数中的 ModuleOp，表示待转换的IR
    // 2. target 表示转换的目标，包含了哪些Dialect是合法的
    // 3. patterns 表示重写规则集合
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace toy

int main(int argc, char **argv) {
  // Diaclect 注册表
  DialectRegistry registry;
  // 注册三种Dialect：toy、arith、func，输入Toy IR, 输出Arith IR
  registry.insert<toy::ToyDialect, arith::ArithDialect, func::FuncDialect>();
  // 自定义的 Pass 注册成可用Pass， 通过命令行参数 lower-toy-to-arith（getArgument()） 来调用
  PassRegistration<toy::LowerToyToArithPass>();
  // 读取main的输入参数，，从registry解析加载Dialect，运行Pass pipeline，输出转化后的IR
  return asMainReturnCode(MlirOptMain(argc, argv, "Toy tutorial driver\n", registry));
}
