// The ODS counterpart to 01_toy-opt.cpp. Operation declarations, accessors,
// parsers, printers, builders, and most verification are generated from .td.

#include "../ods/OdsToyOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "OdsToyOps.cpp.inc"

#include "OdsToyCostOpInterface.cpp.inc"

namespace ods_toy {

int64_t Toy_AddOp::getCost() { return 1; }

int64_t Toy_MultiplyOp::getCost() { return 3; }

class OdsToyDialect : public Dialect {
public:
  explicit OdsToyDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<OdsToyDialect>()) {
    addOperations<Toy_AddOp, Toy_MultiplyOp, Toy_ScaleOp>();
  }

  static StringRef getDialectNamespace() { return "ods_toy"; }
};

class LowerAdd : public OpRewritePattern<Toy_AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Toy_AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, op.lhs(), op.rhs());
    return success();
  }
};

class LowerMultiply : public OpRewritePattern<Toy_MultiplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Toy_MultiplyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.lhs(), op.rhs());
    return success();
  }
};

class LowerScale : public OpRewritePattern<Toy_ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Toy_ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto factor = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(op.factor().convertToFloat()));
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.input(), factor);
    return success();
  }
};

// This pass deliberately receives only Operation*. It knows neither AddOp nor
// MultiplyOp; the interface is the capability-based dispatch boundary.
struct AnnotateOdsToyCostsPass
    : public PassWrapper<AnnotateOdsToyCostsPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "annotate-ods-toy-costs"; }
  StringRef getDescription() const final {
    return "Annotate operations implementing ToyCostOpInterface";
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    auto i64 = IntegerType::get(context, 64);
    getOperation()->walk([&](Operation *operation) {
      if (auto costOp = dyn_cast<ToyCostOpInterface>(operation))
        operation->setAttr("toy_interface.cost",
                           IntegerAttr::get(i64, costOp.getCost()));
    });
  }
};

struct LowerOdsToyToArithPass
    : public PassWrapper<LowerOdsToyToArithPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-ods-toy-to-arith"; }
  StringRef getDescription() const final {
    return "Replace ODS-defined toy arithmetic with arith operations";
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addIllegalDialect<OdsToyDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAdd, LowerMultiply, LowerScale>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace ods_toy

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<ods_toy::OdsToyDialect, arith::ArithmeticDialect>();
  PassRegistration<ods_toy::LowerOdsToyToArithPass>();
  PassRegistration<ods_toy::AnnotateOdsToyCostsPass>();
  return asMainReturnCode(
      MlirOptMain(argc, argv, "ODS Toy tutorial driver\n", registry));
}
