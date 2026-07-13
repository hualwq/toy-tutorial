// ============================================================
// linalg Fusion：用 linalg-fuse pass 融合操作
// ============================================================
// 目标：展示 linalg-fuse pass 如何把 producer 融合到 consumer 的循环内
//
// 与 TVM compute_at 的对应：
//   TVM:  s[producer].compute_at(s[consumer], axis)
//   MLIR: linalg-fuse pass（自动分析 producer-consumer 关系）
//
// 融合的本质：
//   改变中间值的生命周期，减少内存读写
//   融合前：producer 写全局内存 → consumer 读全局内存
//   融合后：producer 的结果留在寄存器/局部缓存
// ============================================================

// ----------------------------------------------------------
// 变换前：ReLU + Matmul（两个独立 op）
// ----------------------------------------------------------
module @before {
  func.func @relu_matmul(
    %A: tensor<64x64xf32>, %B: tensor<64x64xf32>
  ) -> tensor<64x64xf32> {
    // Producer: ReLU(A)
    %relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%A : tensor<64x64xf32>)
    outs(%A : tensor<64x64xf32>) {
    ^bb0(%a: f32, %c: f32):
      %zero = arith.constant 0.0 : f32
      %cmp = arith.cmpf ogt, %a, %zero : f32
      %result = arith.select %cmp, %a, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>

    // Consumer: Matmul(ReLU(A), B)
    %out = tensor.empty() : tensor<64x64xf32>
    %C = linalg.matmul
      ins(%relu, %B : tensor<64x64xf32>, tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>)
      -> tensor<64x64xf32>

    return %C : tensor<64x64xf32>
  }
}

// ----------------------------------------------------------
// linalg-fuse pass 的效果
// ----------------------------------------------------------
// 运行命令：
//   mlir-opt 03_linalg_fusion.mlir \
//     -pass-pipeline='builtin.module(linalg-fuse{operation-fusion=on})' \
//     --mlir-print-ir-after-all 2>&1 | head -60
//
// 效果：
//   ReLU 和 Matmul 被融合成一个 linalg.generic
//   融合后的 op 同时包含 ReLU 和 Matmul 的逻辑
//   不需要分配 relu 的临时张量
//
// 融合后的计算逻辑（简化）：
//   for i, j:
//     for k:
//       a = A[i][k]
//       a_relu = max(a, 0)   ← ReLU 融合进来了
//       C[i][j] += a_relu * B[k][j]
//
// 注意：linalg-fuse 只能融合满足特定条件的 op：
//   1. producer 的 output 是 consumer 的 input
//   2. 融合后不会产生循环携带依赖
//   3. 通常是 element-wise → reduction 的模式
// ----------------------------------------------------------

// ----------------------------------------------------------
// 手写融合后的等价代码（memref 层）
// ----------------------------------------------------------
module @after_fusion {
  func.func @relu_matmul_fused(
    %A: memref<64x64xf32>,
    %B: memref<64x64xf32>,
    %C: memref<64x64xf32>
  ) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        memref.store %zero, %C[%i, %j] : memref<64x64xf32>
        scf.for %k = %c0 to %c64 step %c1 {
          // 融合点：ReLU 在 matmul 循环内完成
          %a = memref.load %A[%i, %k] : memref<64x64xf32>
          %relu_a = arith.maxf %a, %zero : f32   // ← ReLU 融合进来
          %b = memref.load %B[%k, %j] : memref<64x64xf32>
          %old = memref.load %C[%i, %j] : memref<64x64xf32>
          %prod = arith.mulf %relu_a, %b : f32
          %new = arith.addf %old, %prod : f32
          memref.store %new, %C[%i, %j] : memref<64x64xf32>
        }
      }
    }
    return
  }
}

// ----------------------------------------------------------
// 更复杂的融合：Conv + BatchNorm + ReLU
// ----------------------------------------------------------
// 这是实际模型中的常见模式
// linalg-fuse 可以融合多个 op（如果满足融合条件）
module @conv_bn_relu {
  func.func @model(
    %input: tensor<1x3x32x32xf32>,
    %filter: tensor<16x3x3x3xf32>,
    %gamma: tensor<16xf32>,
    %beta: tensor<16xf32>
  ) -> tensor<1x16x30x30xf32> {
    // Step 1: Conv
    %out_conv = tensor.empty() : tensor<1x16x30x30xf32>
    %conv = linalg.conv_2d_nchw_fchw
      ins(%input, %filter : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
      outs(%out_conv : tensor<1x16x30x30xf32>)
      -> tensor<1x16x30x30xf32>

    // Step 2: BatchNorm (简化版：scale + shift)
    %out_bn = tensor.empty() : tensor<1x16x30x30xf32>
    %bn = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d1)>,
        affine_map<(d0, d1, d2, d3) -> (d1)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%conv, %gamma, %beta : tensor<1x16x30x30xf32>, tensor<16xf32>, tensor<16xf32>)
    outs(%out_bn : tensor<1x16x30x30xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %o: f32):
      %scaled = arith.mulf %x, %g : f32
      %result = arith.addf %scaled, %b : f32
      linalg.yield %result : f32
    } -> tensor<1x16x30x30xf32>

    // Step 3: ReLU
    %out_relu = tensor.empty() : tensor<1x16x30x30xf32>
    %relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%bn : tensor<1x16x30x30xf32>)
    outs(%out_relu : tensor<1x16x30x30xf32>) {
    ^bb0(%x: f32, %o: f32):
      %zero = arith.constant 0.0 : f32
      %cmp = arith.cmpf ogt, %x, %zero : f32
      %result = arith.select %cmp, %x, %zero : f32
      linalg.yield %result : f32
    } -> tensor<1x16x30x30xf32>

    return %relu : tensor<1x16x30x30xf32>
  }
}

// ----------------------------------------------------------
// Fusion 的限制
// ----------------------------------------------------------
// 不是所有 op 都能融合：
//
// 1. 融合后不能引入循环携带依赖
//    → element-wise → reduction 可以
//    → reduction → element-wise 通常不行（需要归约完成）
//
// 2. 融合可能增加寄存器压力
//    → 融合太多 op 可能导致 spilling
//
// 3. 某些 op 有特殊的 lowering 路径
//    → 例如 linalg.matmul 可能被直接 lowering 到 BLAS
//    → 融合后可能失去这个优化机会
//
// 实际编译器中，fusion 是 cost model 驱动的：
// 只在融合收益 > 代价时才做

// ============================================================
// 对照 TVM
// ============================================================
// TVM                              MLIR (linalg-fuse)
// ------------------------------------------------
// s[relu].compute_at(s[matmul], k)  linalg-fuse pass（自动分析）
// 手动指定融合位置                  pass 自动决定是否融合
// 需要理解循环结构                  只需描述计算语义
//
// 关键直觉：
//   linalg 的融合是基于结构化 op 的
//   编译器知道 ReLU 是 element-wise，matmul 是 reduction
//   可以自动判断融合是否合法
//   TVM 需要用户手动指定 compute_at 的位置
