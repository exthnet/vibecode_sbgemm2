# SIMD📁 `ChangeLog.md`
🤖PG1.1
- **ハードウェア**：玄界 Node Group A（1コア）
- **モジュール**：GCC 13.3.1

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載

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
