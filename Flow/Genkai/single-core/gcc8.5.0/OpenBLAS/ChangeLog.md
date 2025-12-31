# OpenBLAS `ChangeLog.md`
PG1.10
- **ハードウェア**：Genkai single-core （1コア）
- **モジュール**：GCC 8.5.0 + OpenBLAS

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載（PMがテンプレートから生成）

---

### v1.0.0
**変更点**: "OpenBLAS cblas_sgemm利用版を実装"
**結果**: コンパイル・実行待ち `- GFLOPS`
**コメント**: "BF16->FP32変換後にcblas_sgemmを呼び出す実装。SSH接続確立待ち"

<details>

- **生成時刻**: `2025-12-31T15:18:34Z`
- [ ] **compile**
    - status: `pending`
    - message: "SSH接続待ち"
- [ ] **job**
    - id: `-`
    - resource_group: `a-batch-low`
    - status: `pending`
- [ ] **test**
    - status: `pending`
    - performance: `-`
    - unit: `GFLOPS`

</details>

