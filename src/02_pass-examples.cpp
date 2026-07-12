// Four small passes for studying the common MLIR pass implementation styles.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace toy {

// These operations intentionally use generic assembly. The example focuses on
// pass mechanics, so their shape metadata is stored in simple ArrayAttr values.
class TransposeOp : public Op<TransposeOp, OpTrait::OneOperand,
                              OpTrait::OneResult, OpTrait::ZeroRegions> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "toy.transpose"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static StringRef names[] = {"permutation"};
    return names;
  }

  static void build(OpBuilder &builder, OperationState &state,
                    Type resultType, Value input, ArrayAttr permutation) {
    state.addOperands(input);
    state.addTypes(resultType);
    state.addAttribute("permutation", permutation);
  }

  Value getInput() { return getOperand(); }
  Value getOutput() { return (*this)->getResult(0); }
  ArrayAttr getPermutation() {
    return (*this)->getAttrOfType<ArrayAttr>("permutation");
  }
};

class ReshapeOp : public Op<ReshapeOp, OpTrait::OneOperand, OpTrait::OneResult,
                            OpTrait::ZeroRegions> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "toy.reshape"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static StringRef names[] = {"shape"};
    return names;
  }

  static void build(OpBuilder &builder, OperationState &state,
                    Type resultType, Value input, ArrayAttr shape) {
    state.addOperands(input);
    state.addTypes(resultType);
    state.addAttribute("shape", shape);
  }

  Value getInput() { return getOperand(); }
  Value getOutput() { return (*this)->getResult(0); }
  ArrayAttr getShape() { return (*this)->getAttrOfType<ArrayAttr>("shape"); }
};

class PassExamplesDialect : public Dialect {
public:
  explicit PassExamplesDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<PassExamplesDialect>()) {
    addOperations<TransposeOp, ReshapeOp>();
  }

  static StringRef getDialectNamespace() { return "toy"; }
};

static bool getI64Values(ArrayAttr attribute,
                         SmallVectorImpl<int64_t> &values) {
  if (!attribute)
    return false;
  for (Attribute element : attribute) {
    auto integer = element.dyn_cast<IntegerAttr>();
    if (!integer)
      return false;
    values.push_back(integer.getInt());
  }
  return true;
}

static ArrayAttr makeI64ArrayAttr(Builder &builder,
                                  ArrayRef<int64_t> values) {
  SmallVector<Attribute> attributes;
  attributes.reserve(values.size());
  for (int64_t value : values)
    attributes.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attributes);
}

// Example 1: Two transposes can be replaced with one composed transpose. If
// the composition is the identity permutation, both operations disappear.

// pattern是 Pass的组成部分，表示的是内层重写的规则
// OpRewritePattern<TransposeOp>表示只处理 TransposeOp
class FoldConsecutiveTranspose : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // op.getInput() 表示的到当前 transpose 的输入 SSA值，getDefiningOp<TransposeOp>() 表示获取这个输入 SSA值是从哪个operation产生的
    auto previous = op.getInput().getDefiningOp<TransposeOp>();
    if (!previous)
      return failure();

    SmallVector<int64_t> first; //对应previous，先执行的转置
    SmallVector<int64_t> second; //对应op，后执行的转置
    // getI64Values(previous.getPermutation(), first) 表示获取 previous 的 permutation 属性值，例如[1, 0, 2]，并存到 first中
    if (!getI64Values(previous.getPermutation(), first) ||
        !getI64Values(op.getPermutation(), second) ||
        first.size() != second.size())
      return failure();

    SmallVector<int64_t> composed;
    // 预分配空间
    composed.reserve(first.size());
    // 遍历second的维度编号
    for (int64_t index : second) {
      // 检查索引大小合法吗
      if (index < 0 || static_cast<size_t>(index) >= first.size())
        return failure();
      composed.push_back(first[index]);
    }

    // 检查是否是恒等变换，如果是恒等变换就直接把两个transpose都去掉
    bool isIdentity = true;
    for (auto pair : llvm::enumerate(composed))
      isIdentity &= pair.index() == static_cast<size_t>(pair.value());

    if (isIdentity)
      rewriter.replaceOp(op, previous.getInput());
    else
      rewriter.replaceOpWithNewOp<TransposeOp>(
          op, op.getOutput().getType(), previous.getInput(),
          makeI64ArrayAttr(rewriter, composed));

    if (previous->use_empty())
      rewriter.eraseOp(previous);
    return success();
  }
};

// Lower the declarative permutation into an element-wise tensor program. For
// each output dimension d, inputIndex[permutation[d]] = outputIndex[d].
class LowerTransposeToTensor : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = op.getInput().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getOutput().getType().dyn_cast<RankedTensorType>();
    SmallVector<int64_t> permutation;
    if (!inputType || !resultType ||
        !getI64Values(op.getPermutation(), permutation) ||
        inputType.getRank() != resultType.getRank() ||
        permutation.size() != static_cast<size_t>(inputType.getRank()))
      return failure();

    SmallVector<bool> seen(permutation.size(), false);
    for (int64_t dimension : permutation) {
      if (dimension < 0 || dimension >= inputType.getRank() ||
          seen[dimension])
        return failure();
      seen[dimension] = true;
    }

    // tensor.generate receives a value for every dynamic result dimension.
    SmallVector<Value> dynamicExtents;
    for (int64_t outputDim = 0; outputDim < resultType.getRank(); ++outputDim) {
      if (resultType.isDynamicDim(outputDim))
        dynamicExtents.push_back(rewriter.create<tensor::DimOp>(
            op.getLoc(), op.getInput(), permutation[outputDim]));
    }

    auto generate = rewriter.create<tensor::GenerateOp>(
        op.getLoc(), resultType, dynamicExtents,
        [&](OpBuilder &builder, Location location, ValueRange outputIndices) {
          SmallVector<Value> inputIndices(inputType.getRank());
          for (auto pair : llvm::enumerate(permutation))
            inputIndices[pair.value()] = outputIndices[pair.index()];
          Value element = builder.create<tensor::ExtractOp>(
              location, op.getInput(), inputIndices);
          builder.create<tensor::YieldOp>(location, element);
        });
    rewriter.replaceOp(op, generate.getResult());
    return success();
  }
};

struct EliminateConsecutiveTransposePass
    : PassWrapper<EliminateConsecutiveTransposePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateConsecutiveTransposePass)

  StringRef getArgument() const final {
    return "eliminate-consecutive-transposes";
  }
  StringRef getDescription() const final {
    return "Compose adjacent toy.transpose operations";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConsecutiveTranspose>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

// Example 1b: Materialize the actual data movement represented by
// toy.transpose. Later tensor/scf/LLVM lowering passes can compile this
// tensor.generate program into loops and target-specific code.
struct LowerToyTransposeToTensorPass
    : PassWrapper<LowerToyTransposeToTensorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToyTransposeToTensorPass)

  StringRef getArgument() const final { return "lower-to-tensor-transpose"; }
  StringRef getDescription() const final {
    return "Lower toy.transpose to tensor.generate and tensor.extract";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerTransposeToTensor>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

// Example 2: The output shape of the latter reshape completely describes the
// combined operation. Point it directly at the input of the former reshape.
class FoldConsecutiveReshape : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto previous = op.getInput().getDefiningOp<ReshapeOp>();
    if (!previous || !op.getShape())
      return failure();

    if (previous.getInput().getType() == op.getOutput().getType())
      rewriter.replaceOp(op, previous.getInput());
    else
      rewriter.replaceOpWithNewOp<ReshapeOp>(
          op, op.getOutput().getType(), previous.getInput(), op.getShape());

    if (previous->use_empty())
      rewriter.eraseOp(previous);
    return success();
  }
};

struct EliminateConsecutiveReshapePass
    : PassWrapper<EliminateConsecutiveReshapePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateConsecutiveReshapePass)

  StringRef getArgument() const final { return "eliminate-consecutive-reshapes"; }
  StringRef getDescription() const final {
    return "Fold adjacent toy.reshape operations";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConsecutiveReshape>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

// Example 3: A wrapper pass can compose an existing MLIR pass. The standard
// inliner understands func.call and func.func and performs the IR splicing.
struct InlineFunctionsPass
    : PassWrapper<InlineFunctionsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineFunctionsPass)

  StringRef getArgument() const final { return "inline-functions"; }
  StringRef getDescription() const final {
    return "Run MLIR's standard function inliner";
  }

  void runOnOperation() final {
    PassManager pipeline(&getContext());
    pipeline.addPass(createInlinerPass());
    if (failed(pipeline.run(getOperation())))
      signalPassFailure();
  }
};

// Example 4: A local shape-inference pass. It uses operand types plus shape
// metadata to refine result types. Production inference also handles dynamic
// dimensions, constraints between ops, and function signatures.
struct InferToyShapesPass
    : PassWrapper<InferToyShapesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferToyShapesPass)

  StringRef getArgument() const final { return "infer-toy-shapes"; }
  StringRef getDescription() const final {
    return "Infer toy.transpose and toy.reshape result shapes";
  }

  void runOnOperation() final {
    bool failedInference = false;
    getOperation()->walk([&](Operation *operation) {
      if (auto transpose = dyn_cast<TransposeOp>(operation)) {
        auto inputType = transpose.getInput().getType().dyn_cast<RankedTensorType>();
        SmallVector<int64_t> permutation;
        if (!inputType || !getI64Values(transpose.getPermutation(), permutation) ||
            permutation.size() != static_cast<size_t>(inputType.getRank())) {
          failedInference = true;
          return;
        }

        SmallVector<int64_t> resultShape;
        resultShape.reserve(permutation.size());
        for (int64_t index : permutation) {
          if (index < 0 || index >= inputType.getRank()) {
            failedInference = true;
            return;
          }
          resultShape.push_back(inputType.getDimSize(index));
        }
        transpose.getOutput().setType(
            RankedTensorType::get(resultShape, inputType.getElementType()));
        return;
      }

      if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
        auto inputType = reshape.getInput().getType().dyn_cast<RankedTensorType>();
        SmallVector<int64_t> resultShape;
        if (!inputType || !getI64Values(reshape.getShape(), resultShape)) {
          failedInference = true;
          return;
        }
        for (int64_t dimension : resultShape) {
          if (dimension < 0) {
            failedInference = true;
            return;
          }
        }
        reshape.getOutput().setType(
            RankedTensorType::get(resultShape, inputType.getElementType()));
      }
    });
    if (failedInference)
      signalPassFailure();
  }
};

} // namespace toy

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<toy::PassExamplesDialect, arith::ArithDialect,
                  func::FuncDialect, tensor::TensorDialect>();

  PassRegistration<toy::EliminateConsecutiveTransposePass>();
  PassRegistration<toy::LowerToyTransposeToTensorPass>();
  PassRegistration<toy::EliminateConsecutiveReshapePass>();
  PassRegistration<toy::InlineFunctionsPass>();
  PassRegistration<toy::InferToyShapesPass>();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "MLIR pass implementation examples\n", registry));
}
