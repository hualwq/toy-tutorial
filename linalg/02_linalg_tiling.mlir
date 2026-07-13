// ============================================================
// linalg Tiling：用 linalg-tile pass 做分块
// ============================================================
// 目标：展示 linalg-tile pass 如何对结构化 op 做自动分块
//
// 与手写 scf.for 的区别：
//   手写 scf.for：已经丢失了结构化信息（不知道是 matmul）
//   linalg-tile：在结构化 op 上操作，保留语义信息
//
// 变换路径：
//   linalg.matmul → linalg-tile → 带 tile 的 linalg.matmul（仍是结构化）
//                  → linalg-lower-to-loops → scf.for 循环
// ============================================================

// ----------------------------------------------------------
// 变换前：一个 128×256×192 的 matmul
// ----------------------------------------------------------
module @before {
  func.func @matmul(
    %A: tensor<128x256xf32>,
    %B: tensor<256x192xf32>
  ) -> tensor<128x192xf32> {
    %out = tensor.empty() : tensor<128x192xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<128x256xf32>, tensor<256x192xf32>)
      outs(%out : tensor<128x192xf32>)
      -> tensor<128x192xf32>
    return %C : tensor<128x192xf32>
  }
}

// ----------------------------------------------------------
// linalg-tile pass 的效果
// ----------------------------------------------------------
// 运行命令：
//   mlir-opt 02_linalg_tiling.mlir \
//     -pass-pipeline='builtin.module(linalg-tile{loop-type=scf sizes=64,64})' \
//     -o /dev/stdout
//
// 效果：
//   linalg.matmul 被拆成 2×3 = 6 个 tile
//   每个 tile 仍是一个 linalg.matmul（形状变小）
//   外层是 scf.for 循环，内层仍是结构化 op
//
// 输出 IR 大致结构（简化）：
//   scf.for %io = 0 to 128 step 64 {
//     scf.for %jo = 0 to 192 step 64 {
//       %tile_A = tensor.extract_slice %A[%io, 0][64, 256][1, 1]
//       %tile_B = tensor.extract_slice %B[0, %jo][256, 64][1, 1]
//       %tile_C = tensor.extract_slice %out[%io, %jo][64, 64][1, 1]
//       linalg.matmul ins(%tile_A, %tile_B) outs(%tile_C)
//     }
//   }
//
// 关键：tile 后内层仍是 linalg.matmul！
// 这意味着可以继续做 fusion / vectorization
// ----------------------------------------------------------

// ----------------------------------------------------------
// 完整 pipeline：tile → lower → bufferize
// ----------------------------------------------------------
// mlir-opt 02_linalg_tiling.mlir \
//   -pass-pipeline='builtin.module(
//     linalg-tile{loop-type=scf sizes=64,64},
//     linalg-lower-to-loops{loop-type=scf},
//     convert-linalg-to-affine-loops,
//     affine-loop-fusion,
//     scf-for-loop-canonicalization
//   )' \
//   -o /dev/stdout

// ----------------------------------------------------------
// 手写等价结果（tile + lower 后）
// 这是 mlir-opt 自动生成的等价代码
// ----------------------------------------------------------
module @after_tile_lower {
  func.func @matmul_tiled(
    %A: memref<128x256xf32>,
    %B: memref<256x192xf32>,
    %C: memref<128x192xf32>
  ) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c192 = arith.constant 192 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    // tile(64,64) 后的循环结构
    scf.for %io = %c0 to %c128 step %c64 {
      scf.for %jo = %c0 to %c192 step %c64 {
        // 初始化 tile 内的 C 块
        scf.for %ii = %c0 to %c64 step %c1 {
          scf.for %ji = %c0 to %c64 step %c1 {
            %i = arith.addi %io, %ii : index
            %j = arith.addi %jo, %ji : index
            memref.store %zero, %C[%i, %j] : memref<128x192xf32>
          }
        }
        // 归约循环（k 维）
        scf.for %k = %c0 to %c256 step %c1 {
          scf.for %ii = %c0 to %c64 step %c1 {
            scf.for %ji = %c0 to %c64 step %c1 {
              %i = arith.addi %io, %ii : index
              %j = arith.addi %jo, %ji : index
              %a = memref.load %A[%i, %k] : memref<128x256xf32>
              %b = memref.load %B[%k, %j] : memref<256x192xf32>
              %old = memref.load %C[%i, %j] : memref<128x192xf32>
              %prod = arith.mulf %a, %b : f32
              %new = arith.addf %old, %prod : f32
              memref.store %new, %C[%i, %j] : memref<128x192xf32>
            }
          }
        }
      }
    }
    return
  }
}

// ----------------------------------------------------------
// 不同 loop-type 的效果对比
// ----------------------------------------------------------
// 1. loop-type=scf（默认）
//    → 生成 scf.for，通用循环
//
// 2. loop-type=affine
//    → 生成 affine.for，可做依赖分析
//    → 适合做 affine-loop-fusion 等 pass
//
// 3. loop-type=parallel
//    → 生成 scf.parallel，表示外层可并行
//    → 后端可映射到 OpenMP / CUDA block
//
// 例：用 affine 循环做 tiling
//   mlir-opt 02_linalg_tiling.mlir \
//     -pass-pipeline='builtin.module(
//       linalg-tile{loop-type=affine sizes=64,64},
//       convert-linalg-to-affine-loops
//     )' -o /dev/stdout

// ============================================================
// 对照 TVM
// ============================================================
// TVM                          MLIR (linalg-tile)
// ------------------------------------------------
// s[C].tile(i, j, 64, 64)      linalg-tile{loop-type=scf sizes=64,64}
// tile 后仍是 TE 计算            tile 后仍是 linalg 结构化 op
// 需要 apply() 才展开            linalg-lower-to-loops 才展开
//
// 关键直觉：
//   linalg 层做 tiling 比 scf 层更"经济"
//   因为结构化 op 包含语义信息（matmul、conv 等）
//   pass 可以根据语义做更智能的分块
//   例如：知道是 matmul 后，可以自动选择 tile 大小
