// ============================================================
// linalg Vectorization：从结构化 op 到 SIMD 向量
// ============================================================
// 目标：展示 linalg op 如何 lowering 到 vector dialect
//
// 向量化路径（两种）：
//   路径 A：linalg → scf → vector（先展开循环，再向量化）
//   路径 B：linalg → vector.transfer（直接在 linalg 层向量化）
//
// 本文件主要展示路径 A（更通用）
// 路径 B 需要更专门的 pass（如 linalg-vectorize）
// ============================================================

// ----------------------------------------------------------
// 变换前：逐元素加法（linalg.generic）
// ----------------------------------------------------------
module @before {
  func.func @add(
    %A: tensor<128x128xf32>,
    %B: tensor<128x128xf32>
  ) -> tensor<128x128xf32> {
    %out = tensor.empty() : tensor<128x128xf32>
    %C = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%out : tensor<128x128xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<128x128xf32>
    return %C : tensor<128x128xf32>
  }
}

// ----------------------------------------------------------
// 完整向量化 pipeline
// ----------------------------------------------------------
// 步骤：
//   1. bufferize：tensor → memref（分配内存）
//   2. tile：做分块（例如 tile 到 8×8）
//   3. lower-to-loops：linalg → scf.for
//   4. vectorize：scf → vector（用 vector.transfer_read/write）
//   5. lower-vector：vector → LLVM IR
//
// 运行命令（完整 pipeline）：
//   mlir-opt 04_linalg_vectorization.mlir \
//     -pass-pipeline='builtin.module(
//       one-shot-bufferize,
//       linalg-tile{loop-type=scf sizes=8,8},
//       linalg-lower-to-loops{loop-type=scf},
//       scf-for-loop-canonicalization,
//       convert-scf-to-cf,
//       vectorize,
//       vector-lowering
//     )' \
//     -o /dev/stdout 2>&1 | head -100

// ----------------------------------------------------------
// 手写向量化后的等价代码
// ----------------------------------------------------------
module @after_vectorize {
  func.func @add_vectorized(
    %A: memref<128x128xf32>,
    %B: memref<128x128xf32>,
    %C: memref<128x128xf32>
  ) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index

    // tile(8,8) 后的循环
    scf.for %io = %c0 to %c128 step %c8 {
      scf.for %jo = %c0 to %c128 step %c8 {
        // 向量化：一次读/算/写 8 个元素
        scf.for %ii = %c0 to %c8 step %c1 {
          %i = arith.addi %io, %ii : index
          %j_base = arith.muli %jo, %c1 : index  // jo=0, so j_base=0

          // vector.transfer_read：从内存读取 8 个连续 f32
          // 对应 TVM 的 A[ramp(j, 1, 8)]
          %vec_a = vector.transfer_read %A[%i, %j_base], %c1
            {in_bounds = [true, true]}
            : memref<128x128xf32>, vector<8xf32>

          %vec_b = vector.transfer_read %B[%i, %j_base], %c1
            {in_bounds = [true, true]}
            : memref<128x128xf32>, vector<8xf32>

          // 向量加法（一次操作 8 个 f32）
          %vec_sum = arith.addf %vec_a, %vec_b : vector<8xf32>

          // vector.transfer_write：写回内存
          vector.transfer_write %vec_sum, %C[%i, %j_base], %c1
            {in_bounds = [true, true]}
            : vector<8xf32>, memref<128x128xf32>
        }
      }
    }
    return
  }
}

// ----------------------------------------------------------
// 向量化与 tiling 的关系
// ----------------------------------------------------------
// 向量化通常需要先 tiling：
//   - 向量长度是固定的（如 8 个 f32 = 256 bit）
//   - tiling 确保内层循环大小是向量长度的整数倍
//   - 这样向量化才是合法的（不会越界）
//
// 例：tile(8,8) 后，内层循环大小 = 8
//     正好可以用 vector<8xf32> 做向量化
//
// 如果内层循环大小不是 8：
//   → 需要做 "masked vectorization"（用 mask 处理尾部）
//   → 或者做 "peeling"（把尾部单独处理）

// ----------------------------------------------------------
// vector dialect 的核心操作
// ----------------------------------------------------------
// 1. vector.transfer_read
//    从 memref 读取连续元素到向量寄存器
//    可以指定 in_bounds 来消除边界检查
//
// 2. vector.transfer_write
//    从向量寄存器写回 memref
//
// 3. arith.addf / mulf 等
//    对 vector<Nxf32> 做逐元素操作
//    一次操作 N 个 f32
//
// 4. vector.broadcast
//    标量 → 向量（把所有元素设为同一个值）
//
// 5. vector.extract / vector.insert
//    从向量中提取/插入元素
//
// 6. vector.contract
//    向量级矩阵乘法（比逐元素操作更高级）

// ----------------------------------------------------------
// 更高级的向量化：vector.contract
// ----------------------------------------------------------
// linalg.matmul 可以直接 lowering 到 vector.contract
// 这是比逐元素向量化更高级的优化
//
// vector.contract 对应：
//   C[i][j] += A[i][k] * B[k][j]  （矩阵乘法模式）
//
// 例（简化）：
//   %va = vector.transfer_read %A[%i, %k_base] ...
//   %vb = vector.transfer_read %B[%k_base, %j] ...
//   %vc = vector.contract %va, %vb, %vc_init
//           {indexing_maps = ...}
//     : vector<4x4xf32>, vector<4x4xf32> -> vector<4x4xf32>
//
// vector.contract 可以被后端直接映射为：
//   - AVX2:  _mm256_fmadd_ps
//   - AVX512: _mm512_fmadd_ps
//   - ARM SVE: svmla_f32

// ============================================================
// 对照 TVM
// ============================================================
// TVM                              MLIR (vector)
// ------------------------------------------------
// s[C].vectorize(yi)               vector.transfer_read/write
// TVM TIR: ramp(base, 1, 8)       vector<8xf32> 作为 SSA value
// 向量化在 TIR 层做               linalg → vector 的 lowering 路径
//
// 关键直觉：
//   TVM 的 vectorize 是在 TIR 层面将 loop 改写为 ramp
//   MLIR 的 vector dialect 把向量操作作为显式的 SSA value
//   两者都暴露了 SIMD 并行性给编译器后端
//
//   MLIR 的优势：vector dialect 是通用的
//   可以表示任意向量长度和操作
//   TVM 的 ramp 更贴近 LLVM 的向量 IR
