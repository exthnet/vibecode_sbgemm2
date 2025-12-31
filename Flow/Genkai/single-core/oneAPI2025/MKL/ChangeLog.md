# MKL📁 `ChangeLog.md`
🤖PG1.9
- **ハードウェア**: Genkai single-core (a-batch-low)
- **モジュール**: Intel oneAPI 2025.1.3 (MKL sbgemm)

## Change Log

- 基本の型: `ChangeLog_format.md`に記載
- PMオーバーライド: なし

---

### v1.1.0
**変更点**: "詳細ベンチマーク機能追加（複数サイズ自動テスト、統計計算、CSV出力）"
**結果**: コード作成完了（実行待ち）
**コメント**: "9サイズ（128〜4096）の自動ベンチマーク、標準偏差・帯域幅計測を追加"

<details>

- **生成時刻**: `2025-12-31T15:22:15Z`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - id: `未投入`
    - resource_group: `a-batch-low`
    - status: `pending`
- [ ] **test**
    - status: `pending`
    - performance: `TBD`
    - unit: `GFLOPS`
- **params**:
    - matrix_sizes: `128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096`
    - csv_output: `benchmark_results.csv`

</details>

### v1.0.0
**変更点**: "MKL cblas_sbgemm を使用した BF16 行列積の初期実装"
**結果**: コード作成完了（実行待ち）
**コメント**: "SSH接続不可のためコンパイル・実行は保留中"

<details>

- **生成時刻**: `2025-12-31T15:14:52Z`
- [ ] **compile**
    - status: `pending`
    - log: `未実行`
- [ ] **job**
    - id: `未投入`
    - resource_group: `a-batch-low`
    - status: `pending`
- [ ] **test**
    - status: `pending`
    - performance: `TBD`
    - unit: `GFLOPS`
- **params**:
    - matrix_sizes: `256x256, 1024x1024, 2048x2048`
    - iterations: `5-10`
    - mkl_threads: `1`

</details>

