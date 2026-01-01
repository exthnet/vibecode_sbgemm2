# SIMD📁 `ChangeLog.md`
🤖PG1.1
- **ハードウェア**：玄界 Node Group A（1コア）
- **モジュール**：GCC 13.3.1

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載

### v1.2.0
**変更点**: "AVX-256インターリーブパッキング、AVX-512 C転送最適化"
**結果**: 理論性能の32.6%達成 `633.69 GFLOPS`
**コメント**: "最大サイズで1%改善。中間サイズで性能低下あり、パッキングオーバーヘッド要改善"

<details>

- **生成時刻**: `2026-01-01T09:56:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `compile_v1.2.0.log`
- [x] **job**
    - id: `4610366`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:56:10Z`
    - end_time: `2026-01-01T09:56:35Z`
    - runtime_sec: `25`
    - status: `success`
- [x] **test**
    - performance: `633.69`
    - unit: `GFLOPS`
    - efficiency: `32.6%`
- [x] **sota**
    - scope: `local`
- **params**:
    - block_k: `1536`
    - block_n: `480`
    - kernel: `Tiling_B 2x3 + AVX-256 pack`
- **results_by_size**:
    - 256: `290.82 GFLOPS (14.9%)`
    - 512: `544.93 GFLOPS (28.0%)`
    - 1024: `583.14 GFLOPS (30.0%)`
    - 2048: `590.14 GFLOPS (30.3%)`
    - 4096: `559.99 GFLOPS (28.8%)`
    - 8192: `487.80 GFLOPS (25.1%)`
    - 10000: `633.69 GFLOPS (32.6%)`

</details>

### v1.1.0
**変更点**: "Tiling_B 2x3カーネル実装、AVX-512プリフェッチ追加"
**結果**: 理論性能の32.2%達成 `627.33 GFLOPS`
**コメント**: "v1.0.1から12%改善。目標65%に向け継続最適化"

<details>

- **生成時刻**: `2026-01-01T09:50:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `compile_v1.1.0.log`
- [x] **job**
    - id: `4610359`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:50:10Z`
    - end_time: `2026-01-01T09:50:32Z`
    - runtime_sec: `22`
    - status: `success`
- [x] **test**
    - performance: `627.33`
    - unit: `GFLOPS`
    - efficiency: `32.2%`
- [x] **sota**
    - scope: `local`
- **params**:
    - block_k: `1536`
    - block_n: `480`
    - kernel: `Tiling_B 2x3`
- **results_by_size**:
    - 256: `167.55 GFLOPS (8.6%)`
    - 512: `319.05 GFLOPS (16.4%)`
    - 1024: `494.80 GFLOPS (25.4%)`
    - 2048: `523.72 GFLOPS (26.9%)`
    - 4096: `585.86 GFLOPS (30.1%)`
    - 8192: `584.70 GFLOPS (30.1%)`
    - 10000: `627.33 GFLOPS (32.2%)`

</details>

### v1.0.1
**変更点**: "タイルレジスタ番号を定数に修正、2x2カーネル実装"
**結果**: 理論性能の28.8%達成 `560.31 GFLOPS`
**コメント**: "v1.0.0のコンパイルエラー修正。目標65%に対し28.8%、改善余地あり"

<details>

- **生成時刻**: `2026-01-01T09:46:01Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `compile_v1.0.1.log`
- [x] **job**
    - id: `4610350`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:46:26Z`
    - end_time: `2026-01-01T09:46:50Z`
    - runtime_sec: `24`
    - status: `success`
- [x] **test**
    - performance: `560.31`
    - unit: `GFLOPS`
    - efficiency: `28.8%`
- **params**:
    - block_k: `1536`
    - block_n: `480`
    - kernel: `2x2 tiles`
- **results_by_size**:
    - 256: `132.06 GFLOPS (6.8%)`
    - 512: `274.23 GFLOPS (14.1%)`
    - 1024: `431.47 GFLOPS (22.2%)`
    - 2048: `467.35 GFLOPS (24.0%)`
    - 4096: `531.80 GFLOPS (27.3%)`
    - 8192: `528.66 GFLOPS (27.2%)`
    - 10000: `560.31 GFLOPS (28.8%)`

</details>

### v1.0.0
**変更点**: "Intel AMX Tiling_B手法初期実装"
**結果**: コンパイルエラー（タイルレジスタ番号が変数）
**コメント**: "reference.pdfのTiling_B手法（k=1536, n=480）を実装試行。AMX intrinsicsの制約によりエラー"

<details>

- **生成時刻**: `2026-01-01T09:40:00Z`
- [ ] **compile**
    - status: `error`
    - message: "bad register name `%tmmtj' - タイル番号が変数のためアセンブラエラー"
    - log: `compile_v1.0.0.log`
- [ ] **job**
    - status: `not_submitted`

</details>
