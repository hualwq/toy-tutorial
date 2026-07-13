# linalg 示例：结构化线性代数变换

本目录包含 `linalg` dialect 的 tiling、fusion、vectorization 示例，展示如何用 MLIR 的 pass pipeline 做自动变换。

## 文件列表

| 文件 | 主题 | 对应 TVM 概念 |
|------|------|--------------|
| `01_linalg_basics.mlir` | linalg.generic / matmul / conv 基础语法 | `te.compute` |
| `02_linalg_tiling.mlir` | linalg-tile pass 做分块 | `s[C].tile()` |
| `03_linalg_fusion.mlir` | linalg-fuse pass 做融合 | `s[prod].compute_at()` |
| `04_linalg_vectorization.mlir` | linalg → vector lowering | `s[C].vectorize()` |

## 运行方式

### 前置条件

需要 LLVM/MLIR 构建（含 `mlir-opt`）：
```bash
# 检查 mlir-opt 是否可用
command -v mlir-opt
mlir-opt --version
```

如果未安装，参考 `../../CLAUDE.md` 中的 MLIR 构建说明。

### 验证 IR 合法性

```bash
# 验证所有示例文件
for f in toy-tutorial/linalg/*.mlir; do
  echo "=== $f ==="
  mlir-opt "$f" --verify-each 2>&1 | head -5
done
```

### 运行 pass pipeline

#### 例 1：linalg-tile（tiling）

```bash
mlir-opt toy-tutorial/linalg/02_linalg_tiling.mlir \
  -pass-pipeline='builtin.module(linalg-tile{loop-type=scf sizes=64,64})' \
  -o /dev/stdout
```

#### 例 2：linalg-fuse（fusion）

```bash
mlir-opt toy-tutorial/linalg/03_linalg_fusion.mlir \
  -pass-pipeline='builtin.module(linalg-fuse{operation-fusion=on})' \
  --mlir-print-ir-after-all 2>&1 | head -80
```

#### 例 3：完整 pipeline（tile → lower → vectorize）

```bash
mlir-opt toy-tutorial/linalg/04_linalg_vectorization.mlir \
  -pass-pipeline='builtin.module(
    one-shot-bufferize,
    linalg-tile{loop-type=scf sizes=8,8},
    linalg-lower-to-loops{loop-type=scf},
    scf-for-loop-canonicalization
  )' \
  -o /dev/stdout 2>&1 | head -100
```

## 核心概念

### linalg 的结构化 op

```
linalg.matmul / linalg.conv_2d / linalg.generic
    ↓
indexing_maps: 描述张量访问模式
iterator_types: 描述迭代语义（parallel / reduction）
    ↓
保持结构化信息 → 方便做 tiling / fusion / vectorization
```

### 变换路径

```
tensor 层（值语义）
    ↓ one-shot-bufferize
memref 层（内存语义）
    ↓ linalg-tile / linalg-fuse
带变换的结构化 op
    ↓ linalg-lower-to-loops
scf.for 循环
    ↓ vectorize
vector.transfer_read/write
    ↓ lower-vector
LLVM IR → 机器码
```

## 与 TVM 的对照

| TVM | MLIR (linalg) |
|-----|--------------|
| `te.compute` | `linalg.generic` / `linalg.matmul` |
| `s[C].tile(i, j, 64, 64)` | `linalg-tile{loop-type=scf sizes=64,64}` |
| `s[prod].compute_at(s[cons], k)` | `linalg-fuse` pass |
| `s[C].vectorize(yi)` | `vectorize` pass → `vector.transfer_*` |

## 学习建议

1. 先读 `01_linalg_basics.mlir`，理解 linalg 的基础语法
2. 用 `mlir-opt --verify-each` 验证每个文件
3. 用 `--mlir-print-ir-after-all` 观察 pass 前后的 IR 变化
4. 对比 `mlir-examples/03_tile.mlir` 等文件，理解手写循环 vs pass 自动变换的区别
