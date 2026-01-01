# Unroll📁 `ChangeLog.md`
🤖PG1.4
- **ハードウェア**：玄界 Node Group A （1コア）
- **モジュール**：GCC 13.3.1

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載（PMがテンプレートから生成）

---

### v1.4.0
**変更点**: "ループ交換(k-i-j順序) + 4x4タイル"
**結果**: 理論性能の0.06%達成 `1.14 GFLOPS`
**コメント**: "大幅低下。Cへの書き込み頻度増加でキャッシュ効率悪化。i-j-kが最適"

<details>

- **生成時刻**: `2026-01-01T10:00:00Z`
- [x] **compile**
    - status: `success`
    - options: `-O3 -march=native`
- [x] **job**
    - id: `4610374`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:59:58Z`
    - end_time: `2026-01-01T10:00:18Z`
    - runtime_sec: `20`
    - status: `success`
- [x] **test**
    - performance: `1.14`
    - unit: `GFLOPS`
    - efficiency: `0.06%`
    - avg_time: `1.877693 sec`

</details>

---

### v1.2.1
**変更点**: "v1.2.0にfunroll-loopsオプション追加"
**結果**: 理論性能の0.53%達成 `10.35 GFLOPS`
**コメント**: "v1.2.0より低下。funroll-loopsは逆効果"

<details>

- **生成時刻**: `2026-01-01T09:56:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - options: `-O3 -march=native -funroll-loops`
- [x] **job**
    - id: `4610364`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:55:27Z`
    - end_time: `2026-01-01T09:55:29Z`
    - runtime_sec: `2`
    - status: `success`
- [x] **test**
    - performance: `10.35`
    - unit: `GFLOPS`
    - efficiency: `0.53%`
    - avg_time: `0.207403 sec`

</details>

---

### v1.3.0
**変更点**: "8x8タイル + __builtin_prefetch + restrict修飾子"
**結果**: 理論性能の0.54%達成 `10.53 GFLOPS`
**コメント**: "v1.2.0より若干低下。8x8タイルはレジスタスピル発生の可能性。4x4が最適"

<details>

- **生成時刻**: `2026-01-01T09:54:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - options: `-O3 -march=native -funroll-loops`
- [x] **job**
    - id: `4610362`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:53:40Z`
    - end_time: `2026-01-01T09:53:42Z`
    - runtime_sec: `2`
    - status: `success`
- [x] **test**
    - performance: `10.53`
    - unit: `GFLOPS`
    - efficiency: `0.54%`
    - avg_time: `0.203903 sec`
- **params**:
    - M: `1024`
    - N: `1024`
    - K: `1024`
    - iterations: `10`
    - tile_i: `8`
    - tile_j: `8`

</details>

---

### v1.2.0
**変更点**: "4x4タイルブロッキングによるレジスタ最大活用"
**結果**: 理論性能の0.56%達成 `10.84 GFLOPS`
**コメント**: "v1.1.0比71%向上。コンパイラの自動SIMD化が効果的"

<details>

- **生成時刻**: `2026-01-01T09:50:53Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - options: `-O3 -march=native`
- [x] **job**
    - id: `4610360`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:50:10Z`
    - end_time: `2026-01-01T09:50:12Z`
    - runtime_sec: `2`
    - status: `success`
- [x] **test**
    - performance: `10.84`
    - unit: `GFLOPS`
    - efficiency: `0.56%`
    - avg_time: `0.198120 sec`
- [x] **sota**
    - scope: `local`
- **params**:
    - M: `1024`
    - N: `1024`
    - K: `1024`
    - iterations: `10`
    - tile_i: `4`
    - tile_j: `4`

</details>

---

### v1.1.0
**変更点**: "2x4タイルブロッキング + K8倍展開によるレジスタ最適化"
**結果**: 理論性能の0.32%達成 `6.31 GFLOPS`
**コメント**: "v1.0.0比6倍向上。レジスタ再利用によりメモリアクセス削減"

<details>

- **生成時刻**: `2026-01-01T09:47:34Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - options: `-O3 -march=native`
- [x] **job**
    - id: `4610351`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:47:01Z`
    - end_time: `2026-01-01T09:47:04Z`
    - runtime_sec: `3`
    - status: `success`
- [x] **test**
    - performance: `6.31`
    - unit: `GFLOPS`
    - efficiency: `0.32%`
    - avg_time: `0.340413 sec`
- [x] **sota**
    - scope: `local`
- **params**:
    - M: `1024`
    - N: `1024`
    - K: `1024`
    - iterations: `10`
    - tile_i: `2`
    - tile_j: `4`
    - unroll_k: `8`

</details>

---

### v1.0.0
**変更点**: "Kループ4倍展開によるループアンローリング最適化"
**結果**: 理論性能の0.05%達成 `1.04 GFLOPS`
**コメント**: "スカラーコードのため低性能。次はレジスタブロッキング追加を検討"

<details>

- **生成時刻**: `2026-01-01T09:42:20Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `compile_v1.0.0.log`
    - options: `-O3 -march=native`
- [x] **job**
    - id: `4610346`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:42:36Z`
    - end_time: `2026-01-01T09:42:58Z`
    - runtime_sec: `22`
    - status: `success`
- [x] **test**
    - performance: `1.04`
    - unit: `GFLOPS`
    - efficiency: `0.05%`
    - avg_time: `2.060187 sec`
- **params**:
    - M: `1024`
    - N: `1024`
    - K: `1024`
    - iterations: `10`
    - unroll_factor: `4`

</details>
