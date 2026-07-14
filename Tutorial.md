# 从零读懂一个最小 Toy 方言

本目录把官方 MLIR Toy 教程压缩为一个可以逐文件阅读的最小实验。它只处理
`f32` 标量加法和乘法，目标是先看清 **方言、operation、pass 和 lowering** 的
关系。完整 Toy 教程随后会加入词法/语法分析、张量、shape inference、ODS 和 LLVM
代码生成。

## 先回答：这个例子解决什么问题？

假设 Toy 源程序写的是：

```toy
def scaled_sum(x, y, scale) {
  return (x + y) * scale;
}
```

前端最初希望保留它自己的操作名：`toy.add`、`toy.multiply`。这样可以在高层
阶段施加 Toy 自己的约束或优化。后端却不应认识每一种前端语言，因此稍后把它们
转换成通用的 `arith.addf` 和 `arith.mulf`。

```text
Toy source
    -> toy.add / toy.multiply       (Toy 的高层语义)
    -> arith.addf / arith.mulf      (可复用的标量算术语义)
    -> 后续的 LLVM dialect / LLVM IR
```

这正是 MLIR 方言解决的问题：不同 IR 词汇可以在同一基础设施中定义、验证、混用
和逐步转换；MLIR 不要求整个程序从一开始就使用同一种 IR。

## 文件地图

| 文件 | 读它时要回答的问题 |
| --- | --- |
| `input/01_toy_high_level.mlir` | 前端为什么要使用 `toy.*`，而不立刻使用 `arith.*`？ |
| `src/01_toy-opt.cpp` | 如何注册方言、定义 operation、编写 conversion pass？ |
| `src/02_pass-examples.cpp` | 如何用四种常见技巧实现 optimization / inference pass？ |
| `ods/OdsToyOps.td` | 如何用 ODS 声明 dialect、operand、result、attribute、trait、builder 和 assembly format？ |
| `ods/OdsToyOps.h` + `src/04_ods-toy-opt.cpp` | 如何 include TableGen 产物、注册 ODS operation，并使用生成的 accessor？ |
| `ods/OdsToyCostOpInterface.td` | 如何声明自定义 OpInterface，使不同 operation 暴露同一项能力？ |
| `input/09_ods_operations.mlir` | ODS operation 的自定义文本语法、默认属性和 lowering 是什么样子？ |
| `src/05_pass-manager-greedy.cpp` | PassManager 如何按 IR 层级调度 pass，GreedyPatternRewrite 如何迭代至不再变化？ |
| `input/10_pass_manager_greedy.mlir` | 一个可观察到多轮贪心重写、且包含两个 function 的最小输入。 |
| `input/02_after_lowering.mlir` | lowering 后哪些东西保留，哪些东西改变？ |
| `input/03_mixed_dialects.mlir` | 一个 module 能否同时拥有多个方言？ |
| `CMakeLists.txt` | 怎样把这个最小 driver 连接到本地 MLIR 构建？ |

## 第 0 步：准备工具

本仓库不携带 MLIR 源码或二进制。请先依照仓库根目录的 `Learn.md` 构建 MLIR，确保
有一个包含 `MLIRConfig.cmake` 的构建目录。以下假设该目录是 `$MLIR_BUILD`：

```sh
cmake -S mlir-examples/toy-tutorial -B mlir-examples/toy-tutorial/build \
  -DMLIR_DIR="$MLIR_BUILD/lib/cmake/mlir"
cmake --build mlir-examples/toy-tutorial/build
```

构建成功后，教程驱动在 `build/toy-opt`。它相当于一个只额外认识 `toy` 方言和
`-lower-toy-to-arith` pass 的小型 `mlir-opt`。

## 第 1 步：只读高层 Toy IR

打开 `input/01_toy_high_level.mlir`，先不要运行 pass：

```mlir
%sum = "toy.add"(%x, %y) : (f32, f32) -> f32
%result = "toy.multiply"(%sum, %scale) : (f32, f32) -> f32
```

每条 operation 都有三个关键部分：操作名、operand 和 result type。`%sum` 是 SSA
value：它只定义一次，`toy.multiply` 通过引用它建立数据依赖。

这里采用 MLIR 的通用（generic）语法，目的是让 operand 与结果类型一眼可见。实际
的官方 Toy 教程会进一步为 operation 定义更易读的自定义语法。

运行验证和打印：

```sh
mlir-examples/toy-tutorial/build/toy-opt \
  mlir-examples/toy-tutorial/input/01_toy_high_level.mlir --verify-each
```

问题：为什么标准 `mlir-opt` 通常不能读这份文件？答案是它没有注册本教程定义的
`ToyDialect`，因此不知道 `toy.add` 的 operation 契约。

## 第 2 步：读方言定义

在 `src/01_toy-opt.cpp` 中依次读 `AddOp`、`MultiplyOp` 和 `ToyDialect`。

```cpp
static StringRef getOperationName() { return "toy.add"; }
```

这行把 C++ 类和文本 IR 的 `toy.add` 名字连接起来。`NOperands<2>`、`OneResult` 和
`ZeroRegions` 是 operation 的结构约束：它需要两个输入、有一个结果、没有嵌套的
region。`ToyDialect` 构造函数中的 `addOperations<...>()` 则把这些 operation 注册到
`toy` 命名空间。

注意这个例子故意没有定义专用的 `ToyType`。它复用 `f32`，说明方言可以复用其他
方言或 builtin 提供的类型；方言不等于“一套必须完全自给自足的 IR”。

## 第 3 步：执行 lowering

运行：

```sh
mlir-examples/toy-tutorial/build/toy-opt \
  mlir-examples/toy-tutorial/input/01_toy_high_level.mlir \
  -lower-toy-to-arith --verify-each
```

输出应与 `input/02_after_lowering.mlir` 一致。为了逐个观察 rewrite，可加：

```sh
mlir-examples/toy-tutorial/build/toy-opt \
  mlir-examples/toy-tutorial/input/01_toy_high_level.mlir \
  -lower-toy-to-arith --mlir-print-ir-after-all
```

`LowerAdd` 与 `LowerMultiply` 各是一个局部 rewrite：匹配 Toy operation，再以相同
operand 创建 `arith` operation，最后用新结果替代旧结果。因为浮点加法/乘法的语义
在此例中对应，数据流没有改变。

`LowerToyToArithPass` 里的 `ConversionTarget` 是 conversion 的完成标准：`toy`
方言被标记为 illegal，`arith` 与 `func` 被标记为 legal。所以当 pass 完成后仍有一个
`toy.*` operation，转换就失败。这比“尽量改写一些 operation”更严格，也更适合
真正的 lowering pipeline。

## 第 4 步：观察方言混用

运行：

```sh
mlir-examples/toy-tutorial/build/toy-opt \
  mlir-examples/toy-tutorial/input/03_mixed_dialects.mlir \
  -lower-toy-to-arith --verify-each
```

输入同时使用：

| 方言 | 本例操作 | 职责 |
| --- | --- | --- |
| `func` | `func.func`、`return` | 函数边界与控制结构 |
| `toy` | `toy.add`、`toy.multiply` | Toy 前端的计算词汇 |
| `arith` | `arith.constant` | 可复用的标量常量 |

这就是“方言不绑定单一 IR 层级”的具体证据：同一个 module 可以在同一时刻混合多个
方言；lowering 时只替换其中一部分操作。这里 `func` 和 `arith.constant` 原样留下，
只有 `toy.*` 被消除。

## 下一步：怎样走向官方完整 Toy？

1. 把当前只接受 `f32` 的 verifier 扩展为“两个 operand 与 result 类型必须一致”，
   并支持更多浮点类型。
2. 用 ODS/TableGen 重写 operation 定义；仓库的 `mlir-ods/` 是下一站。
3. 将标量改为 `tensor<...xf32>`，增加 `toy.transpose`，观察 shape 推导为什么需要
   独立 pass。
4. 将 `toy` lowering 到 `linalg`/`tensor`，再经过 bufferization、`scf`/`memref` 和
   LLVM dialect。这时就接近官方 Toy 教程的完整路线。

每一步都应先写输入 IR 和预期输出 IR，再写 pass。这样你验证的是语义变化，而不是只
验证 C++ 能编译。

## 第 5 步：用 ODS 定义 Operation

`src/01_toy-opt.cpp` 的 `AddOp` 和 `MultiplyOp` 是手写的，适合先理解 `Op` 类的
基本结构；本节则使用 `ods/OdsToyOps.td` 定义等价的 `ods_toy.add` 和
`ods_toy.multiply`。两份代码并列存在，方便逐项对照，而不是替换前面的手写教学。

先构建专用 driver：

```sh
cmake --build mlir-examples/toy-tutorial/build --target ods-toy-opt
```

`CMakeLists.txt` 中四个关键动作依次是：设置 `LLVM_TARGET_DEFINITIONS` 为 `.td`
文件、调用 `mlir_tablegen(... -gen-op-decls)` 与
`mlir_tablegen(... -gen-op-defs)`、用 `add_public_tablegen_target` 聚合生成依赖、让
`ods-toy-opt` 依赖该 target。生成文件位于 `build/OdsToyOps.h.inc` 与
`build/OdsToyOps.cpp.inc`，不应手改；修改 `.td` 后重新构建即可生成。

### 5.1 怎样通过代码学习 ODS

不要先试图记住全部 TableGen 语法。以 `ods_toy.add` 为一条主线，按下面的生成链路
往返阅读代码：

```text
input/09_ods_operations.mlir
  ods_toy.add %x, %y : f32
            |
            v
ods/OdsToyOps.td
  ODS_Toy_AddOp: arguments / results / traits / assemblyFormat
            |
            | mlir-tblgen（由 CMake 调用）
            v
build/OdsToyOps.h.inc + build/OdsToyOps.cpp.inc
  Toy_AddOp 类、lhs()/rhs()/result()、build()、verify()、parse()/print()
            |
            v
src/04_ods-toy-opt.cpp
  addOperations<Toy_AddOp>()、OpRewritePattern<Toy_AddOp>、op.lhs()
```

具体阅读时，先在 `input/09_ods_operations.mlir` 找到 operation 的文本形态，再在
`OdsToyOps.td` 中定位同名 mnemonic。只看五个字段：`Op<...>` 表示 operation 属于哪
个 dialect，`arguments`/`results` 决定 SSA 输入输出，traits 提供跨值约束，
`builders` 决定额外 C++ 构建入口，`assemblyFormat` 决定易读 IR 语法。

然后打开构建目录的生成文件，搜索 `class Toy_AddOp` 与 `Toy_AddOp::verify`。不要逐行
阅读所有生成代码，只确认 `.td` 中每一个名字都落到了哪里：`$lhs` 对应 `lhs()`，
`F32` 对应类型检查，`SameOperandsAndResultType` 对应 operation-level verifier，
`assemblyFormat` 对应 `parse()` 和 `print()`。最后回到 `04_ods-toy-opt.cpp`，观察
lowering 不再通过 `getOperand(0)` 访问输入，而是直接使用生成的 `op.lhs()`/`op.rhs()`。

每次只改一个声明并重建，再比较生成差异和打印出的 IR。例如把 `F32` 临时改成
`AnyType`，观察 verifier 缺少的约束；或给 `scale` 显式写入 `factor`，观察
`attr-dict` 如何打印。这样学习的是“字段如何改变 C++ API 和 IR”，而不是孤立记忆
ODS 名称。

#### 5.1.1 这个示例里的 `dag` 是什么

TableGen 的 `dag` 是 **DAG（有向无环图）的一个节点初始化器**。它用一个操作符和一组
指向子节点的边来描述节点，语法为：

```tablegen
(operator argument0, argument1, ...)
```

这里的 `operator` 是 TableGen record，`argumentN` 是它的子节点；argument 自身也可以
是另一个 `dag`，因此嵌套后形成有向无环图。`:$name` 是给一条边的目标命名，供 ODS
生成 C++ API 使用。

在 ODS 中，`dag` 是声明 operation 结构的基本表示。当前示例只使用了 **单层 DAG**，且
每个子节点只出现一次，因此看起来很像“操作符加参数列表”；这只是 DAG 的一个树形特例，
并不表示 `dag` 不是图。示例有三处真实用法：

```tablegen
// 1. Op 的 argument dag：ins 是操作符；每一项是一个 operand 或 attribute。
let arguments = (ins F32:$input,
  DefaultValuedAttr<F32Attr, "1.0f">:$factor);

// 2. Op 的 result dag：outs 是操作符；每一项是一个 result。
let results = (outs F32:$result);

// 3. 自定义 builder 的参数 dag：ins 是操作符；每一项是 C++ 参数类型和名字。
OpBuilder<(ins "::mlir::Value":$input, "float":$factor), [{
  // builder implementation
}]>
```

可以把第一段读成一个有两条出边的节点：根节点是 `ins`，第一条边指向
`F32:$input`，第二条边指向 `DefaultValuedAttr<...>:$factor`。`$input` 和 `$factor`
是给边的目标取的名字；ODS 以它们生成 C++ accessor、builder 参数和 attribute 名字。
`outs` 同理，只是它描述 result。`OpBuilder` 复用相同的 `dag` 语法，但子节点不再是
IR 约束，而是普通 C++ 函数参数。

因此，读 ODS 时先找每个 `dag` 的根操作符：`ins` 表示输入侧，`outs` 表示输出侧；再
看每个带 `$` 的子节点，它就是随后在 C++ API 与 assembly format 中会出现的名字。

按下面顺序阅读 `OdsToyOps.td`：

1. `ODS_ToyDialect` 给方言指定 IR 名字 `ods_toy` 与 C++ namespace。
2. `ODS_ToyOp` 是三个 operation 的公共 TableGen 基类。
3. `ODS_Toy_AddOp` 的 `arguments`、`results` 和命名的 `$lhs`/`$rhs` 会生成
   `lhs()`、`rhs()`、`result()` accessor 和多个 `build(...)` 重载；`F32` 与
   `SameOperandsAndResultType` 共同生成 verifier。
4. `assemblyFormat` 将 generic IR 变成 `ods_toy.add %x, %y : f32` 这样的紧凑语法，
   并自动生成 parser 和 printer。
5. `ODS_Toy_ScaleOp` 展示 `DefaultValuedAttr<F32Attr, "1.0f">`。省略 `factor`
   时，生成的 `factor()` accessor 返回默认值；它的自定义 `OpBuilder` 还展示了如何
   以原始 `float` 参数构建 attribute。

### 5.2 运行解析和 verifier

```sh
mlir-examples/toy-tutorial/build/ods-toy-opt \
  mlir-examples/toy-tutorial/input/09_ods_operations.mlir --verify-each
```

输入中的 `ods_toy.scale %product : f32` 没有显式 attribute，因而使用 `1.0f`。
要显式指定 factor，可写成：

```mlir
%result = ods_toy.scale %product {factor = 2.0 : f32} : f32
```

### 5.3 观察 ODS operation 的 lowering

接着运行 `-lower-ods-toy-to-arith`：

```sh
mlir-examples/toy-tutorial/build/ods-toy-opt \
  mlir-examples/toy-tutorial/input/09_ods_operations.mlir \
  -lower-ods-toy-to-arith --verify-each
```

`04_ods-toy-opt.cpp` 中的 rewrite 使用 ODS 自动生成的 `lhs()`、`rhs()`、`input()`
和 `factor()`。这正是 ODS 的直接收益：operation 的结构性代码由 `.td` 统一生成，
C++ 只留下方言注册和本例真正关心的 lowering 语义。

## 第 6 步：用 OpInterface 解耦通用 Pass

这个实验使用自定义的 `ToyCostOpInterface`。它不是为了真实地预测硬件耗时，而是用一个
小而可观察的 `getCost()` 契约，展示接口如何让 Pass 依赖“能力”而不是某个具体
operation 类。

```text
Toy_AddOp       -- implements getCost() = 1 --\
                                             --> ToyCostOpInterface --> 通用 Pass
Toy_MultiplyOp  -- implements getCost() = 3 --/
Toy_ScaleOp     -- does not implement it ---  跳过
```

### 6.1 从接口定义读起

打开 `ods/OdsToyCostOpInterface.td`。`OpInterface<"ToyCostOpInterface">` 定义 C++
接口名，`InterfaceMethod<..., "int64_t", "getCost", (ins)>` 声明接口方法。它本身不
规定任何 operand、result 或 attribute；它只要求“实现该接口的 operation 能回答
`getCost()`”。

`CMakeLists.txt` 对这份 `.td` 运行 `-gen-op-interface-decls` 和
`-gen-op-interface-defs`，生成 `build/OdsToyCostOpInterface.h.inc` 与 `.cpp.inc`。接口
必须单独放在 `.td` 文件中：若与引用多个标准接口的 operation 定义混在一起，接口生成器
会同时重新生成那些标准接口的声明。

### 6.2 看 operation 如何声明与实现接口

在 `ods/OdsToyOps.td` 中，`add` 与 `multiply` 的 trait 列表包含：

```tablegen
DeclareOpInterfaceMethods<ODS_ToyCostOpInterface>
```

它让 ODS 在对应 Op 类中声明 `getCost()`，但不提供具体语义。具体实现留在
`src/04_ods-toy-opt.cpp`：`Toy_AddOp::getCost()` 返回 `1`，
`Toy_MultiplyOp::getCost()` 返回 `3`。`scale` 没有该 trait，也没有该方法，因此它不实现
接口。这是接口与普通 helper 函数的区别：调用方可在运行时先判断一个任意 `Operation *`
是否具有该能力。

### 6.3 运行通用接口 Pass

```sh
cmake --build mlir-examples/toy-tutorial/build --target ods-toy-opt

mlir-examples/toy-tutorial/build/ods-toy-opt \
  mlir-examples/toy-tutorial/input/09_ods_operations.mlir \
  -annotate-ods-toy-costs --verify-each
```

输出中只有 `ods_toy.add` 与 `ods_toy.multiply` 会获得
`toy_interface.cost = ... : i64`，`ods_toy.scale` 保持不变。阅读
`AnnotateOdsToyCostsPass::runOnOperation()` 时，关键是这两行：

```cpp
if (auto costOp = dyn_cast<ToyCostOpInterface>(operation))
  operation->setAttr("toy_interface.cost", ... costOp.getCost() ...);
```

Pass 不写 `dyn_cast<Toy_AddOp>` 或 `dyn_cast<Toy_MultiplyOp>`；新增一个实现
`ToyCostOpInterface` 的 operation 后，Pass 无需修改。这就是 OpInterface 的价值。

Trait 与 Interface 的关系是：Trait 通常是 Op 类上的静态性质或结构约束，例如
`SameOperandsAndResultType`；Interface 是可以从 `Operation *` 动态查询的能力契约，带有
可调用方法。ODS 中 `DeclareOpInterfaceMethods<...>` 同时作为 trait 写在 operation 定义里，
但它的语义是“该 Op 实现接口”，不是另一个普通约束。

可以继续练习：为 `Toy_ScaleOp` 加上该 interface trait 和 `getCost()` 实现，重建后运行
同一命令，确认通用 Pass 自动开始标注 `scale`，而无需改动 Pass。

## 第 7 步：分清 PassManager 与 GreedyPatternRewrite

这两个概念经常一起出现，但它们解决的不是同一层问题：

```text
PassManager
  决定：哪些 pass、以什么顺序、作用于哪一层 IR operation
  例：顶层 ModuleOp -> 每个嵌套 ModuleOp -> GreedyAddZeroCleanupPass

GreedyPatternRewrite
  决定：在一个 pass 的作用范围内，局部 rewrite rule 如何反复匹配 IR
  例：arith.addf x, 0.0  ->  x，直到再也找不到可改写处
```

构建新 driver：

```sh
cmake --build mlir-examples/toy-tutorial/build --target pass-manager-greedy
```

再运行：

```sh
mlir-examples/toy-tutorial/build/pass-manager-greedy \
  mlir-examples/toy-tutorial/input/10_pass_manager_greedy.mlir \
  -run-toy-cleanup-pipeline --verify-each
```

输出中的两个嵌套 module 最终都为空。这里有两层“重复”，不要混淆：
`PassManager` 的 `nest<ModuleOp>()` 对顶层 module 中的**每个嵌套 module 各运行一次** cleanup
pass；而 cleanup pass 内的 `applyPatternsAndFoldGreedily` 对**同一个 module 的 IR**反复运行
pattern，直至达到 fixed point。

### 9.1 PassManager 是编译 pipeline 的调度器

看 `RunToyCleanupPipelinePass::runOnOperation()`：

```cpp
PassManager pipeline(&getContext());
pipeline.nest<ModuleOp>().addPass(
    std::make_unique<GreedyAddZeroCleanupPass>());
pipeline.run(getOperation());
```

顶层 `pipeline` 的 op 类型是 `builtin.module`；`nest<ModuleOp>()` 建立了一个只处理嵌套
`builtin.module` 的子 pipeline。因此输入有两个嵌套 module 时，`GreedyAddZeroCleanupPass`
的 `runOnOperation()` 被调用两次，每次得到一个子 `ModuleOp`。这就是 pass 的 operation
scope：不同层级的 operation 可以拥有不同 pass。真实编译器通常会把这一层替换为
`func.func`；本示例保留 builtin module，以便直接观察嵌套 ModuleOp 的调度行为。
PassManager 负责顺序、嵌套、失败传播和 IR 验证选项，但它不定义 `x + 0`
如何优化。

命令行里的 `-run-toy-cleanup-pipeline` 是一个包装 pass；它演示在 C++ 中构建 pipeline。
也可以将 pass pipeline 直接交给 `mlir-opt` 风格的命令行解析器。工程实践中，前者适合把
一串稳定的项目 pipeline 封成一个名字，后者适合实验、调试和改变 pass 顺序。

### 9.2 GreedyPatternRewrite 是局部规则的固定点迭代器

`FoldAddZero` 是一个 `OpRewritePattern<arith::AddFOp>`。它只匹配浮点加法，若任一 operand
匹配常量零，便调用 `rewriter.replaceOp(op, otherOperand)`：旧 operation 被移除，旧 result
的所有 SSA use 自动换成另一个 operand。

`@three_rewrite_rounds` 的数据依赖是：

```text
%0 = x  + 0    -> x
%1 = %0 + 0    -> x
%2 = %1 + 0    -> x
```

每次替换都会让后续 operation 的输入发生变化，因而可能创造新的匹配机会。`greedy` driver
持续尝试 patterns，直到一个完整迭代没有 IR 改动；这就是 fixed point。它不是“挑一个看起来
最优的 rule”，而是“可改则改，直到稳定”。如果两条规则互相把 IR 改回去，就永远不会收敛；
driver 会在达到重写次数上限后失败，示例将该失败传给 `signalPassFailure()`。

`applyPatternsAndFoldGreedily` 比 `applyPatternsGreedily` 多一步：每轮也尝试 operation 的
通用 fold/canonicalization，并会清理本例中因 rewrite 而变为无 use 的常量。因此两个子 module
最终为空。这里的 `FoldAddZero` 是显式 pattern，故即使不依赖这个额外 folding 也能工作。

### 9.3 推荐的阅读顺序

1. 先读 `input/10_pass_manager_greedy.mlir`，确认有两个嵌套 module 和连续三个 `addf`。
2. 再读 `FoldAddZero::matchAndRewrite()`：它只描述一条局部等价变换。
3. 看 `GreedyAddZeroCleanupPass`：pattern 放入 `RewritePatternSet`，随后由 greedy driver
   反复执行。
4. 最后看 `RunToyCleanupPipelinePass`：它用嵌套 `PassManager` 将 cleanup pass 调度到
   顶层 module 内每一个子 module。

为了观察 pass 边界和每轮 IR，可在命令末尾加入：

```sh
--print-ir-before-all --print-ir-after-all
```

对这个示例最值得记住的一句话是：**PassManager 组织 pass；greedy rewrite 在某个 pass
内部组织 patterns。**

## 第 8 步：四种 Pass 实现技巧

`src/02_pass-examples.cpp` 使用独立的 `pass-examples` driver。它定义了最小的
`toy.transpose` 与 `toy.reshape`，把重点放在 Pass 的写法，而不是完整张量 dialect。

`toy.transpose` 本身只是声明式 IR，不会直接执行数据搬运。`-lower-to-tensor-transpose`
展示其实际计算：对输出坐标 `out`，构造输入坐标
`inputIndex[permutation[outDim]] = out[outDim]`，以 `tensor.extract` 读取元素，再由
`tensor.generate` 写入输出张量。后续可继续将 tensor dialect lower 为 `scf` 循环、memref
和 LLVM IR。

建议按下面的顺序学习，每次只执行一个 Pass，并在运行命令后增加
`--mlir-print-ir-after-all` 观察 IR 的变化。

### 7.1 先读注册代码

阅读 `02_pass-examples.cpp` 末尾的 `main`。四个 `PassRegistration<...>()` 分别将
Pass 类注册为 `-eliminate-consecutive-transposes`、
`-eliminate-consecutive-reshapes`、`-inline-functions` 和 `-infer-toy-shapes`。
先确认“命令行参数如何找到 Pass”，再进入具体实现。

### 7.2 学习 Pattern Rewrite：Transpose

打开 `input/04_pass_patterns.mlir`。前两个 `toy.transpose` 都使用 `[1, 0]`：第一次将
`tensor<2x3xf32>` 变成 `tensor<3x2xf32>`，第二次再变回来。运行 transpose Pass 后，
这两个 operation 应被消除。

接着阅读 `FoldConsecutiveTranspose::matchAndRewrite()`：

1. `getDefiningOp<TransposeOp>()` 找到 producer。
2. `getI64Values()` 读取两个 permutation attribute。
3. 组合 permutation；若结果是 identity，用 `rewriter.replaceOp` 替换为原始输入。
4. 否则用 `replaceOpWithNewOp` 创建一个合并后的 transpose。
5. 前一个 transpose 没有 use 时，将其删除。

注意 `EliminateConsecutiveTransposePass::runOnOperation()` 只负责建立
`RewritePatternSet` 并调用 `applyPatternsAndFoldGreedily`；局部语义变化完全写在 pattern
中。这是实现局部优化 Pass 的标准分工。

### 7.3 学习 Pattern Rewrite：Reshape

仍使用 `input/04_pass_patterns.mlir`。后两个 reshape 先变成一维 `tensor<6xf32>`，再
恢复为 `tensor<2x3xf32>`。阅读 `FoldConsecutiveReshape::matchAndRewrite()`，观察它为何
能跳过中间 reshape：后一个 operation 的 `shape` 已完整描述最终结果。

这里要对比两个分支：最终类型与原始输入类型相同时直接删除 reshape；不相同时创建一个
从原始输入直接到最终 shape 的新 reshape。随后将 transpose 与 reshape 两个参数放在同一
条命令中，体会 Pass pipeline 按命令行参数顺序执行。

### 7.4 学习组合标准 Pass：函数内联

打开 `input/05_inline.mlir`，其中 `caller` 通过 `func.call` 调用 `twice`。阅读
`InlineFunctionsPass::runOnOperation()`：它创建局部 `PassManager`，加入
`createInlinerPass()`，再对当前 `ModuleOp` 运行。

这个例子刻意不手写 call graph 分析和 block 拼接，而是展示一个很实用的技巧：把 MLIR
已有能力包装为项目中可命名、可排序、可与自定义 Pass 混用的 Pass。

### 7.5 学习遍历和类型更新：Shape Inference

打开 `input/06_shape_inference.mlir`。结果类型故意是动态 shape，例如
`tensor<?x?xf32>`；但输入的 `RankedTensorType` 和 `permutation`/`shape` attribute 已经
携带了足够的信息。

阅读 `InferToyShapesPass::runOnOperation()`：它用 `walk` 遍历 operation；transpose
按 permutation 重排输入维度，reshape 读取目标 shape；最后通过
`getOutput().setType(...)` 将精确的 `RankedTensorType` 写回 IR。这个 Pass 不替换或删除
operation，说明 Pass 也可以是“分析后细化 IR 元数据”的 transformation。

生产级形状推导还需考虑动态维度、广播、跨 operation 约束和函数签名，但这个例子先聚焦
Pass 的基本控制流与 IR 修改方式。

```sh
# 1. Pattern rewrite：组合相邻 transpose；互逆时直接消除。
mlir-examples/toy-tutorial/build/pass-examples \
  mlir-examples/toy-tutorial/input/04_pass_patterns.mlir \
  -eliminate-consecutive-transposes

# 1b. 物化 transpose 的逐元素计算。对于 permutation = [1, 0]，
# 输出元素 output[i, j] 由输入元素 input[j, i] 取得。
mlir-examples/toy-tutorial/build/pass-examples \
  mlir-examples/toy-tutorial/input/07_transpose_lowering.mlir \
  -lower-to-tensor-transpose

# 2. Pattern rewrite：跳过中间 reshape，直接从原输入 reshape 到最终形状。
mlir-examples/toy-tutorial/build/pass-examples \
  mlir-examples/toy-tutorial/input/04_pass_patterns.mlir \
  -eliminate-consecutive-reshapes

# 可把两个 pattern pass 放进同一个 pipeline。
mlir-examples/toy-tutorial/build/pass-examples \
  mlir-examples/toy-tutorial/input/04_pass_patterns.mlir \
  -eliminate-consecutive-transposes -eliminate-consecutive-reshapes

# 3. 组合已有 Pass：运行标准的 func.call inliner。
mlir-examples/toy-tutorial/build/pass-examples \
  mlir-examples/toy-tutorial/input/05_inline.mlir -inline-functions

# 4. 分析/更新 IR：根据 operand type 与 shape/permutation 属性写回 result type。
mlir-examples/toy-tutorial/build/pass-examples \
  mlir-examples/toy-tutorial/input/06_shape_inference.mlir -infer-toy-shapes
```

前两个例子都从 `OperationPass<ModuleOp>` 的 `runOnOperation()` 建立
`RewritePatternSet`，再调用 `applyPatternsAndFoldGreedily`。pattern 的模板参数决定
它接收哪种 operation；`rewriter.replaceOp...` 自动替换 SSA result 的所有使用处。
Transpose 需要组合两个 permutation；reshape 的最终 `shape` 已经完整描述结果，因此
可以绕过中间 reshape。

内联例子展示另一种技巧：自定义 Pass 不必重写全部算法。`InlineFunctionsPass` 在自己
的 `runOnOperation()` 中创建一个局部 `PassManager`，再加入 MLIR 提供的
`createInlinerPass()`。真实项目中这常用于把标准 pass 和项目专用 pass 组合为稳定的
pipeline。

最后的 shape inference 例子没有 rewrite operation，而是遍历 module，读取输入的
`RankedTensorType` 和 operation attribute，并用 `result.setType(...)` 细化结果类型。
它刻意只处理静态维度，以突出演示重点；生产级推导还要处理动态维度、跨 operation
约束、广播规则，以及函数签名和调用点的一致更新。

## 第 9 步：从 Toy 到 Affine、MemRef 与 LLVM

`src/03_lowering--examples.cpp` 使用 `toy.saxpy` 演示一个完整的可执行 pipeline：
`out[i] = a * x[i] + y[i]`。输入在 `input/08_lowering_examples.mlir`。这里故意要求
`memref<8xf32>` 这样的静态长度，因此 Affine loop 的上界可在编译期确定。

```sh
# Toy -> SCF + MemRef + Arith：通用的结构化循环。
mlir-examples/toy-tutorial/build/lowering-examples \
  mlir-examples/toy-tutorial/input/08_lowering_examples.mlir \
  -lower-toy-saxpy-to-scf-memref --verify-each

# Toy -> Affine + MemRef + Arith：循环界限与下标均为静态仿射形式。
mlir-examples/toy-tutorial/build/lowering-examples \
  mlir-examples/toy-tutorial/input/08_lowering_examples.mlir \
  -lower-toy-saxpy-to-affine-memref --verify-each

# Toy -> SCF -> ControlFlow -> LLVM dialect。
mlir-examples/toy-tutorial/build/lowering-examples \
  mlir-examples/toy-tutorial/input/08_lowering_examples.mlir \
  -lower-toy-saxpy-to-llvm --verify-each

# Toy -> Affine -> ControlFlow -> LLVM dialect。
mlir-examples/toy-tutorial/build/lowering-examples \
  mlir-examples/toy-tutorial/input/08_lowering_examples.mlir \
  -lower-toy-saxpy-affine-to-llvm --verify-each
```

当前项目使用 MLIR 16：SCF 先经 `createConvertSCFToCFPass()` 转为 `cf`，再由项目中的
conversion patterns 将 `arith`、`memref`、`cf` 与 `func` 转为 LLVM dialect。
最终的 LLVM dialect 可见 `llvm.getelementptr`、`llvm.load`、`llvm.fmul`、`llvm.store` 和
`llvm.br`。它仍是 MLIR，不是文本 LLVM IR；如需导出文本 LLVM IR，还可接 `mlir-translate
--mlir-to-llvmir`。


## 第 10 步：读懂 Operation、Region、Block 与 ModuleOp

不要把 operation 只理解为加、减、乘、除等一条计算指令。在 MLIR 中，operation 是
任何具有独立语义的 IR 节点：`arith.addf` 是 operation，`func.func`、`scf.for`、
`scf.if` 和 `builtin.module` 也都是 operation。小 operation 通常只计算一个结果；较大
operation 则可拥有内部 IR。

它们的包含关系是：

```text
Operation
  └─ 0 个或多个 Region
       └─ 1 个或多个 Block
            ├─ 0 个或多个 block argument
            └─ 按顺序排列的 Operation
                 └─ ... 每个 Operation 又可拥有 Region
```

这里的“嵌套”是 **operation 的嵌套**，而不是“MLIR 等于嵌套 module”。`module` 只是
一种常见的顶层 operation；复杂 operation 通常通过 region 容纳内部 IR，而不是通过
嵌套 `ModuleOp`。

### 10.1 ModuleOp 是一种顶层容器 Operation

文本中的：

```mlir
module {
  func.func @add(%x: f32, %y: f32) -> f32 {
    %0 = arith.addf %x, %y : f32
    return %0 : f32
  }
}
```

对应的 IR 树如下：

```text
builtin.module                         Operation，也称 ModuleOp
└─ Region #0                            module 的内部程序
   └─ Block #0                          顶层顺序执行的 operation 容器
      └─ func.func @add                Operation：函数
         └─ Region #0                  函数体
            └─ Block #0                函数入口基本块
               ├─ block arguments: %x: f32, %y: f32
               ├─ arith.addf           Operation：加法
               └─ func.return          Operation：返回
```

`ModuleOp` 是 C++ 中对 `builtin.module` operation 的包装类型。它通常代表一个编译单元，
并拥有一个 region；本教程的 `OperationPass<ModuleOp>` 表示 pass 从这个 module operation
开始处理其内部 IR。

### 10.2 Region 不是代码开始位置，而是 operation 的内部程序

Region 的抽象层级介于“拥有它的 operation”和“内部的 block”之间。它回答的是：
**这个 operation 的 body 在哪里？** 函数体、循环体、条件分支的 then/else 部分，都是
region 的典型例子。文本中的花括号 `{ ... }` 经常呈现一个 region，但 region 是 IR 的
真实结构，而非单纯的语法标记。

例如 `scf.if` 是一个 operation，它通常有两个 region：

```mlir
scf.if %cond {
  "demo.then"() : () -> ()
} else {
  "demo.else"() : () -> ()
}
```

```text
scf.if Operation
├─ then Region
│  └─ Block
│     └─ demo.then Operation
└─ else Region
   └─ Block
      └─ demo.else Operation
```

相应地，`scf.for` 的循环体是一个 region，`func.func` 的函数体是一个 region，
`builtin.module` 的内容也是一个 region。相反，本教程的 `toy.add` 和 `toy.multiply`
使用 `ZeroRegions`，表示它们没有内部程序，是叶子计算 operation。

### 10.3 Block 是 region 内的控制流和顺序执行单位

Block 位于 region 内，包含一串按顺序执行的 operation，并可在开头声明 block argument。
一个简单函数没有跳转，其 region 只有一个 block，因此通常看不见 block 标签：

```mlir
func.func @add(%x: f32, %y: f32) -> f32 {
  %0 = arith.addf %x, %y : f32
  return %0 : f32
}
```

当一个 region 有显式控制流时，才更容易看到多个 block：

```mlir
func.func @choose(%cond: i1) {
  cf.cond_br %cond, ^yes, ^no

^yes:
  "demo.yes"() : () -> ()
  cf.br ^end

^no:
  "demo.no"() : () -> ()
  cf.br ^end

^end:
  return
}
```

`^yes`、`^no` 与 `^end` 是三个 block。`cf.cond_br` 和 `cf.br` 在 block 之间建立
控制流边；每个 block 内的 operation 则自上而下执行。block argument 还可接收前驱 block
传入的 SSA value，在控制流汇合点承担类似传统 SSA `phi` 的角色：

```mlir
^merge(%value: i32):
  "demo.use"(%value) : (i32) -> ()
```

### 10.4 用职责而非名字记忆

| 概念 | 回答的问题 | 典型实例 |
| --- | --- | --- |
| Operation | “这是什么语义单元？” | 加法、函数、循环、module |
| Region | “这个 operation 的内部程序在哪里？” | 函数体、循环体、then/else body |
| Block | “一条控制流路径上顺序执行什么？” | 函数入口块、`^yes` 分支块 |
| ModuleOp | “顶层编译单元用什么 operation 表示？” | `module { ... }` |

最简记忆法是：**operation 是节点，region 是节点拥有的内部代码空间，block 是该空间中的
基本块，ModuleOp 是最常见的顶层 operation。**

## 第 11 步：从 Toy 到 linalg — 结构化线性代数变换

第 4 步提到将 `toy` 继续 lowering 到 `linalg`/`tensor`。linalg（**Lin**ear **Alg**ebra dialect）是 MLIR 中表达张量/矩阵运算的核心方言。它把 matmul、conv、element-wise 等计算表达为 **结构化 op**，让编译器在保持语义信息的前提下做变换，而不是手写循环。

本步使用 `linalg/` 目录下的文件（而非 `mlir-opt` 构建产物），借助标准 `mlir-opt` 观察 linalg 的 pass pipeline。

### 11.1 linalg 的核心思想

对比两种表达矩阵乘法的方式。

**方式 A：手写 scf.for（无结构化信息）**
```mlir
scf.for %i = %c0 to %cN {
  scf.for %j = %c0 to %cM {
    scf.for %k = %c0 to %cK {
      %a = memref.load %A[%i, %k]
      %b = memref.load %B[%k, %j]
      // ...
    }
  }
}
```
编译器只看到三层循环和 load/store，**不知道这是 matmul**，无法应用 matmul 专用优化（如分块策略）。

**方式 B：linalg.matmul（结构化）**
```mlir
%result = linalg.matmul
  ins(%A, %B : tensor<32x64xf32>, tensor<64x48xf32>)
  outs(%out : tensor<32x48xf32>)
```
编译器知道这是 matmul，知道哪些维度是 parallel（i, j）、哪些是 reduction（k），也了解 A/B/C 的访问模式。**结构化信息保留给后续 pass 使用**。

linalg 提供了两端之间的桥梁：用结构化 op 写计算，用 pass pipeline 做变换，最终 lower 到 scf/vector/llvm。

详细语法见 `linalg/01_linalg_basics.mlir`，三种核心 op：

| op | 用途 | 等价于 |
|----|------|--------|
| `linalg.generic` | 通用的结构化 op（用 region 描述计算） | TVM `te.compute` |
| `linalg.matmul` | 矩阵乘法（命名 op） | TVM `te.matmul` |
| `linalg.conv_2d_*` | 卷积（命名 op，含多种布局） | TVM `topi.nn.conv2d` |

`linalg.generic` 是最灵活的形式，通过 `indexing_maps` 和 `iterator_types` 描述任意张量运算；命名 op 是特定运算的快捷方式。

```sh
# 验证所有 linalg 文件
for f in mlir-examples/toy-tutorial/linalg/*.mlir; do
  echo "=== $(basename $f) ==="
  mlir-opt "$f" --verify-each 2>&1 | head -3
done
```

### 11.2 Tiling：用 linalg-tile 做分块

TVM 中 `s[C].tile(i, j, 64, 64)` 把循环拆成 outer+inner 层。linalg 中对等的工具是 `linalg-tile` pass：

```sh
mlir-opt linalg/02_linalg_tiling.mlir \
  -pass-pipeline='builtin.module(linalg-tile{loop-type=scf sizes=64,64})'
```

**变换前**，一个 `linalg.matmul` 表达 128×256×192 的矩阵乘法：

```mlir
%C = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x192xf32>)
                   outs(%out : tensor<128x192xf32>)
```

**变换后**，`linalg-tile{loop-type=scf sizes=64,64}` 产生：

```text
scf.for %io = 0 to 128 step 64 {
  scf.for %jo = 0 to 192 step 64 {
    %tile_A = tensor.extract_slice %A[%io, 0][64, 256][1, 1]
    %tile_B = tensor.extract_slice %B[0, %jo][256, 64][1, 1]
    %tile_C = tensor.extract_slice %out[%io, %jo][64, 64][1, 1]
    linalg.matmul ins(%tile_A, %tile_B) outs(%tile_C)
  }
}
```

关键观察：**tile 后内层仍是 linalg.matmul**，而不是展开的 scf.for。这是 linalg 的设计原则——结构化 op 贯穿变换过程，只在最后阶段 lower 到循环。这也意味着 tile 之后还可以继续做 fusion、vectorization。

`loop-type` 参数控制生成的循环形式：

| loop-type | 生成 | 用途 |
|-----------|------|------|
| `scf`（默认） | `scf.for` | 通用循环 |
| `affine` | `affine.for` | 可做依赖分析的循环 |
| `parallel` | `scf.parallel` | 外层可并行（OpenMP / CUDA） |

如果再加上 `linalg-lower-to-loops` 才能看到手写 4 层循环的等价结果。推荐先只跑 `linalg-tile` 观察 IR 变化，再逐步加后续 pass：

```sh
mlir-opt linalg/02_linalg_tiling.mlir \
  -pass-pipeline='builtin.module(
    linalg-tile{loop-type=scf sizes=64,64},
    linalg-lower-to-loops{loop-type=scf}
  )'
```

### 11.3 Fusion：用 linalg-fuse 融合操作

常见优化模式是 ReLU + Matmul 融合，避免将 ReLU 的中间结果写入主存。TVM 中需要手动指定 `s[relu].compute_at(s[matmul], k)`；linalg 的 `linalg-fuse` pass 可**自动分析** producer-consumer 关系并做融合。

```sh
mlir-opt linalg/03_linalg_fusion.mlir \
  -pass-pipeline='builtin.module(linalg-fuse{operation-fusion=on})' \
  --mlir-print-ir-after-all 2>&1 | head -80
```

变换前的 IR 包含两个独立的 linalg op：

```mlir
// Producer: ReLU(A)
%relu = linalg.generic { ... } ins(%A) outs(%A) { ... }
// Consumer: Matmul(ReLU(A), B)
%C = linalg.matmul ins(%relu, %B) outs(%out)
```

融合后的计算逻辑等价于：

```text
for i:
  for j:
    C[i][j] = 0
    for k:
      a_relu = max(A[i][k], 0)   ← ReLU 融合进 matmul 循环
      C[i][j] += a_relu * B[k][j]
```

注意 `linalg-fuse` 能自动判断融合是否合法，它不是暴力地合并所有 op，而是遵循两个基本原则：
1. **方向**：element-wise → reduction 可以融合（ReLU→Matmul）；反过来通常不行。
2. **合法性**：融合后不能引入循环携带依赖。

`03_linalg_fusion.mlir` 末尾还有一个 Conv + BatchNorm + ReLU 的例子，展示实际模型中的多 op 融合场景。这是更接近生产使用的模式。

### 11.4 Vectorization：从 linalg 到 SIMD 向量

向量化是 tiling 后的自然下一步：tile 出合适大小的内层循环后，用向量操作替代逐元素操作。

```sh
mlir-opt linalg/04_linalg_vectorization.mlir \
  -pass-pipeline='builtin.module(
    one-shot-bufferize,
    linalg-tile{loop-type=scf sizes=8,8},
    linalg-lower-to-loops{loop-type=scf}
  )'
```

变换前后对比如下。

**变换前（linalg.generic）**：
```mlir
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>,
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
  outs(%out : tensor<128x128xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
}
```

**变换后（tile(8,8) + lower 到 vector）**：
```mlir
// tile(8,8) + lower 后，内层 ji 循环被向量化为：
%vec_a = vector.transfer_read %A[%i, %j_base], %c1
  {in_bounds = [true, true]} : memref<128x128xf32>, vector<8xf32>
%vec_b = vector.transfer_read %B[%i, %j_base], %c1
  {in_bounds = [true, true]} : memref<128x128xf32>, vector<8xf32>
%vec_sum = arith.addf %vec_a, %vec_b : vector<8xf32>
vector.transfer_write %vec_sum, %C[%i, %j_base], %c1
  {in_bounds = [true, true]} : vector<8xf32>, memref<128x128xf32>
```

`vector.transfer_read` 替代了逐元素的 `memref.load`；`arith.addf` 在 `vector<8xf32>` 上一次操作 8 个 f32；`vector.transfer_write` 替代逐元素的 `memref.store`。后端可以将其映射为 AVX2（`vmovups` + `vaddps`）、ARM NEON 或 SVE。

`in_bounds = [true, true]` 告诉后端没有越界访问，消除边界检查。这个属性来自 tiling 的静态分析：128 被 8 整除，内层循环不可能越界。

### 11.5 完整 pipeline：tile → fuse → vectorize

实际模型不会只用一个 pass，而是组合使用。典型的 linalg 优化 pipeline 是：

```sh
mlir-opt linalg/03_linalg_fusion.mlir \
  -pass-pipeline='builtin.module(
    one-shot-bufferize,               // tensor → memref
    linalg-tile{loop-type=scf sizes=32,32},  // tiling
    linalg-fuse{operation-fusion=on},         // fusion
    linalg-lower-to-loops{loop-type=scf},     // linalg → scf
    scf-for-loop-canonicalization
  )'
```

对应的变换路径：

```text
linalg.generic / matmul（结构化 op，值语义）
    ↓ one-shot-bufferize
linalg.generic / matmul（结构化 op，内存语义）
    ↓ linalg-tile
tile 后的 linalg op
    ↓ linalg-fuse
融合后的 linalg op（如 ReLU 合并进 matmul）
    ↓ linalg-lower-to-loops
scf.for 循环（已失去结构化信息）
    ↓ vectorize / convert-scf-to-cf / ...,lower-vector
vector / LLVM dialect
```

这里的顺序不是随意定的：**tile 先于 fuse** 是因为把计算切小后再融合，能更精确地分析数据重用；**fuse 先于 lower-to-loops** 是因为结构化 op 上的融合更简单（编译器知道哪些是 parallel、哪些是 reduction），展开为 scf.for 后就丢失了这些信息。

### 11.6 与 TVM schedule 的对照

| TVM | MLIR (linalg) |
|-----|--------------|
| `te.compute` | `linalg.generic` 或命名 op（`linalg.matmul`） |
| `s[C].tile(i, j, 64, 64)` | `linalg-tile{loop-type=scf sizes=64,64}` |
| `s[relu].compute_at(s[matmul], k)` | `linalg-fuse` pass（自动分析） |
| `s[C].vectorize(yi)` | `vectorize` → `vector.transfer_read/write` |
| 用户手动构造 schedule | pass pipeline 声明式描述变换 |
| schedule 在 C++ API 中 | pipeline 在命令行或 `PassManager` 中 |

两者的设计哲学差异是：TVM 要求用户**显式调用 schedule 原语**（`tile`、`compute_at`、`vectorize`），变换立即作用于 IR；linalg 把变换封装为 **pass**，用户通过组合 pass pipeline 来驱动，结构化信息在整个变换链中保留更久，展开的时机更晚。

### 11.7 与 Toy 的衔接

回到本教程的主线——把 `toy` lowering 到 linalg 是自然延伸：

```text
toy.add / toy.multiply（自定义高层方言）
    ↓ 自定义 conversion pass
arith.addf / arith.mulf（可复用标量算术）
    ↓ linalg-genericize / padding / ...
linalg.generic / matmul（结构化线性代数）
    ↓ tile / fuse / vectorize（本步学的内容）
scf / memref / vector（循环和内存操作）
    ↓ lowering
LLVM dialect → LLVM IR → 机器码
```

文件的阅读顺序建议：

1. `linalg/01_linalg_basics.mlir` — 理解 linalg 的基本语法
2. `linalg/02_linalg_tiling.mlir` — tiling 自动变换
3. `linalg/03_linalg_fusion.mlir` — fusion 自动变换
4. `linalg/04_linalg_vectorization.mlir` — 向量化路径

每个文件都是**自包含的**：变换前的 IR 可直接用 `mlir-opt` 验证，变换后的结果从命令输出中观察。建议用 `--mlir-print-ir-after-all` 观察 pass 的逐步变化。理解 linalg 后，再回头读官方 Toy 教程（`https://mlir.llvm.org/docs/Tutorials/Toy/`）的第 5-7 章，会有更完整的认识。
