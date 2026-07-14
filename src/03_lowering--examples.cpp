// A small end-to-end lowering example for MLIR 16.
//
// The input operation has deliberately simple semantics:
//   toy.saxpy(%a, %x, %y, %out)
// means:
//   for i = 0 .. N: out[i] = a * x[i] + y[i]
//
// It is lowered through either SCF or Affine loops, both using MemRef and
// Arith operations.  A final pass demonstrates the rest of the route to the
// LLVM dialect. In MLIR 16 the CFG operations produced between SCF and LLVM
// lowering belong to the ControlFlow dialect.

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace toy {

class SaxpyOp : public Op<SaxpyOp, OpTrait::NOperands<4>::Impl,
                          OpTrait::ZeroResults, OpTrait::ZeroRegions> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "toy.saxpy"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  Value getScale() { return getOperand(0); }
  Value getX() { return getOperand(1); }
  Value getY() { return getOperand(2); }
  Value getOutput() { return getOperand(3); }

  LogicalResult verify() {
    if (!getScale().getType().isF32())
      return emitOpError("expects its first operand to be f32");

    auto xType = getX().getType().dyn_cast<MemRefType>();
    auto yType = getY().getType().dyn_cast<MemRefType>();
    auto outputType = getOutput().getType().dyn_cast<MemRefType>();
    if (!xType || !yType || !outputType || xType.getRank() != 1 ||
        yType.getRank() != 1 || outputType.getRank() != 1 ||
        !xType.getElementType().isF32() || !yType.getElementType().isF32() ||
        !outputType.getElementType().isF32())
      return emitOpError("expects three rank-1 memref<?xf32> operands");

    // The Affine version needs a compile-time trip count. Keeping this
    // constraint makes the contrast with the dynamic SCF version explicit.
    int64_t size = xType.getDimSize(0);
    if (ShapedType::isDynamic(size) || yType.getDimSize(0) != size ||
        outputType.getDimSize(0) != size)
      return emitOpError("expects three memrefs with the same static length");
    return success();
  }
};

class ToyLoweringDialect : public Dialect {
public:
  explicit ToyLoweringDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<ToyLoweringDialect>()) {
    addOperations<SaxpyOp>();
  }

  static StringRef getDialectNamespace() { return "toy"; }
};

static int64_t getStaticLength(SaxpyOp op) {
  return op.getX().getType().cast<MemRefType>().getDimSize(0);
}

// toy.saxpy -> scf.for + memref.load/store + arith.mulf/addf.
class LowerSaxpyToSCF : public OpRewritePattern<SaxpyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SaxpyOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // for loop的索引变量
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upper = rewriter.create<arith::ConstantIndexOp>(loc, getStaticLength(op));
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    //SCF Dialect: Structured Control Flow，提供了更高级的控制流操作，例如条件分支和循环等。其循环更结构化，例如affine能这样表示 affine.for %i = max(0, %N - 8) to min(%M, %N + 8)
    //SCF只能先通过arith.maxsi和arith.minsi来计算出max和min的值，然后再通过scf.for来表示循环
    // 创建scp.for
    rewriter.create<scf::ForOp>(
        loc, zero, upper, step, ValueRange(),
        [&](OpBuilder &builder, Location bodyLoc, Value inductionVar,
            ValueRange) {
          Value x = builder.create<memref::LoadOp>(bodyLoc, op.getX(), inductionVar);
          Value y = builder.create<memref::LoadOp>(bodyLoc, op.getY(), inductionVar);
          Value scaled = builder.create<arith::MulFOp>(bodyLoc, op.getScale(), x);
          Value sum = builder.create<arith::AddFOp>(bodyLoc, scaled, y);
          builder.create<memref::StoreOp>(bodyLoc, sum, op.getOutput(), inductionVar);
          builder.create<scf::YieldOp>(bodyLoc);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

// toy.saxpy -> affine.for + affine.load/store + arith.mulf/addf.
class LowerSaxpyToAffine : public OpRewritePattern<SaxpyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SaxpyOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto loop = rewriter.create<AffineForOp>(loc, 0, getStaticLength(op), 1);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    Value inductionVar = loop.getInductionVar();
    Value x = rewriter.create<AffineLoadOp>(loc, op.getX(), inductionVar);
    Value y = rewriter.create<AffineLoadOp>(loc, op.getY(), inductionVar);
    Value scaled = rewriter.create<arith::MulFOp>(loc, op.getScale(), x);
    Value sum = rewriter.create<arith::AddFOp>(loc, scaled, y);
    rewriter.create<AffineStoreOp>(loc, sum, op.getOutput(), inductionVar);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename Pattern>
static LogicalResult lowerToySaxpy(ModuleOp module, MLIRContext &context) {
  ConversionTarget target(context);
  target.addIllegalDialect<ToyLoweringDialect>();
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                         memref::MemRefDialect, scf::SCFDialect,
                         AffineDialect>();
  RewritePatternSet patterns(&context);
  patterns.add<Pattern>(&context);
  return applyPartialConversion(module, target, std::move(patterns));
}

// MLIR 16 exposes the individual dialect-to-LLVM conversion patterns. Keep
// this small pass local so the tutorial does not depend on an optional
// aggregate `convert-to-llvm` library from a particular MLIR package.
static LogicalResult lowerToLLVM(ModuleOp module, MLIRContext &context) {
  LLVMTypeConverter converter(&context);
  RewritePatternSet patterns(&context);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);

  ConversionTarget target(context);
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           cf::ControlFlowDialect, func::FuncDialect>();
  return applyFullConversion(module, target, std::move(patterns));
}

struct LowerToySaxpyToSCFPass
    : PassWrapper<LowerToySaxpyToSCFPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-toy-saxpy-to-scf-memref"; }
  StringRef getDescription() const final {
    return "Lower toy.saxpy to scf, memref, and arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (failed(lowerToySaxpy<LowerSaxpyToSCF>(getOperation(), getContext())))
      signalPassFailure();
  }
};

struct LowerToySaxpyToAffinePass
    : PassWrapper<LowerToySaxpyToAffinePass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final {
    return "lower-toy-saxpy-to-affine-memref";
  }
  StringRef getDescription() const final {
    return "Lower toy.saxpy to affine, memref, and arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    if (failed(lowerToySaxpy<LowerSaxpyToAffine>(getOperation(), getContext())))
      signalPassFailure();
  }
};

// This pass owns an entire pipeline, so one command exposes every remaining
// lowering stage: toy -> scf/memref/arith -> cf -> llvm.
struct LowerToySaxpyToLLVMPass
    : PassWrapper<LowerToySaxpyToLLVMPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-toy-saxpy-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower toy.saxpy through SCF and ControlFlow to the LLVM dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, cf::ControlFlowDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    PassManager pipeline(&getContext());
    pipeline.addPass(std::make_unique<LowerToySaxpyToSCFPass>());
    pipeline.addPass(createConvertSCFToCFPass());
    if (failed(pipeline.run(getOperation())) ||
        failed(lowerToLLVM(getOperation(), getContext())))
      signalPassFailure();
  }
};

// The same LLVM destination can start from Affine. This makes the historical
// Affine -> ControlFlow -> LLVM part of the pipeline directly observable.
struct LowerToySaxpyAffineToLLVMPass
    : PassWrapper<LowerToySaxpyAffineToLLVMPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final {
    return "lower-toy-saxpy-affine-to-llvm";
  }
  StringRef getDescription() const final {
    return "Lower toy.saxpy through Affine and ControlFlow to LLVM dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    cf::ControlFlowDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    PassManager pipeline(&getContext());
    pipeline.addPass(std::make_unique<LowerToySaxpyToAffinePass>());
    pipeline.addPass(createLowerAffinePass());
    pipeline.addPass(createConvertSCFToCFPass());
    pipeline.addPass(createReconcileUnrealizedCastsPass());
    if (failed(pipeline.run(getOperation())) ||
        failed(lowerToLLVM(getOperation(), getContext())))
      signalPassFailure();
  }
};

} // namespace toy


// 怎么理解 Affine  MemRef  Arithmetic 这几个Dialect
// Affine Dialect：affine.for，表达循环空间迭代， for loop很适合使用仿射变换来表达，能做 loop split，fusion，tile，vectorization等操作
// arith Dialect: 标量值计算的Dialect，提供了算术运算符和比较运算符等操作
// MemRef Dialect: 提供了对多维数组的操作

  // 例如矩阵逐元素加法：

  // for (int i = 0; i < M; ++i)
  //   for (int j = 0; j < N; ++j)
  //     C[i][j] = A[i][j] + B[i][j];

  // 在 MLIR 里通常会拆成：

  // affine.for %i = 0 to %M {
  //   affine.for %j = 0 to %N {
  //     %a = memref.load %A[%i, %j] : memref<?x?xf32>
  //     %b = memref.load %B[%i, %j] : memref<?x?xf32>
  //     %sum = arith.addf %a, %b : f32
  //     memref.store %sum, %C[%i, %j] : memref<?x?xf32>
  //   }
  // }

//SCF Dialect: Structured Control Flow，提供了更高级的控制流操作，例如条件分支和循环等。其循环更结构化，例如affine能这样表示 affine.for %i = max(0, %N - 8) to min(%M, %N + 8)
//SCF只能先通过arith.maxsi和arith.minsi来计算出max和min的值，然后再通过scf.for来表示循环


int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<toy::ToyLoweringDialect, AffineDialect,
                  arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                  LLVM::LLVMDialect>();
  PassRegistration<toy::LowerToySaxpyToSCFPass>();
  PassRegistration<toy::LowerToySaxpyToAffinePass>();
  PassRegistration<toy::LowerToySaxpyToLLVMPass>();
  PassRegistration<toy::LowerToySaxpyAffineToLLVMPass>();
  return asMainReturnCode(
      MlirOptMain(argc, argv, "Toy lowering tutorial driver\n", registry));
}
