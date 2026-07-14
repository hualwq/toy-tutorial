// A minimal example of two distinct layers:
//   PassManager: schedules passes over a hierarchy of operations.
//   GreedyPatternRewrite: repeatedly applies local rewrite rules to an op.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace toy {

// This rule is intentionally tiny. A chain such as ((x + 0) + 0) + 0
class FoldAddZero : public OpRewritePattern<arith::AddFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // 
  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    // 检查右操作数是不是0，是的话就把op替换成左操作数
    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    return failure();
  }
};

// This pass is constrained to builtin.module. It receives one nested module,
// not the enclosing top-level module, when scheduled through a nested manager.
struct GreedyAddZeroCleanupPass
    : PassWrapper<GreedyAddZeroCleanupPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "toy-greedy-add-zero"; }
  StringRef getDescription() const final {
    return "Repeatedly fold arith.addf operations with a zero operand";
  }

  void runOnOperation() final {
    // 将 FoldAddZero 注册到 GreedyAddZeroCleanupPass中
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAddZero>(&getContext());

    // 循环的逻辑在这里， config 是 GreedyRewriteConfig 的实例，最大迭代值使用了默认值
    GreedyRewriteConfig config;
    // 对当前的 ModuleOp 贪心的应用所有的 pattern，反复扫描 IR，直到没有更多匹配（达到不动点）
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  }
};

// This module pass owns a pipeline. `nest<ModuleOp>()` says: for every module
// nested in this module, run the cleanup pass.
struct RunToyCleanupPipelinePass
    : PassWrapper<RunToyCleanupPipelinePass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "run-toy-cleanup-pipeline"; }
  StringRef getDescription() const final {
    return "Run a nested function cleanup pipeline from a module pass";
  }

  void runOnOperation() final {
    // 这个PassManager是局部的，不是全局的
    PassManager pipeline(&getContext());
  //     MLIR 的 IR 是嵌套的、树状的结构。比如一个典型的模块：
  // builtin.module  {              ← 顶层 ModuleOp
  //   builtin.module {             ← 嵌套 ModuleOp（子模块）
  //     func.func @foo() {
  //       %0 = arith.addf %a, %c0
  //       ...
  //     }
  //   }
  //   builtin.module {             ← 另一个嵌套 ModuleOp
  //     func.func @bar() {
  //       %1 = arith.addf %b, %c0
  //       ...
  //     }
  //   }
  // }

    // nest返回的是嵌套管道builder，语义是：对每个嵌套的 ModuleOp，运行 GreedyAddZeroCleanupPass
    // 普通的 Pm， Pm.addPass()，添加Pass，Pass跑在顶层的 ModuleOp 上
    pipeline.nest<ModuleOp>().addPass(
        std::make_unique<GreedyAddZeroCleanupPass>());
    if (failed(pipeline.run(getOperation())))
      signalPassFailure();
  }
};

} // namespace toy

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();

  PassRegistration<toy::GreedyAddZeroCleanupPass>();
  PassRegistration<toy::RunToyCleanupPipelinePass>();
  return asMainReturnCode(MlirOptMain(
      argc, argv, "PassManager and greedy rewrite tutorial driver\n", registry));
}
