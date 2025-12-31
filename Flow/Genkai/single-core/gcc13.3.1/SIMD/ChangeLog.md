# SIMD📁 `ChangeLog.md`
🤖PG1.3
- **ハードウェア**：Genkai single-core（1コア）
- **モジュール**：GCC 13.3.1 (module load gcc-toolset/13)

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載（PMがテンプレートから生成）

---

### v1.1.1
**変更点**: "AVX2 SIMD最適化（FMA使用、bf16→fp32変換最適化）"
**結果**: 準備中（SSH接続待ち）
**コメント**: "8要素並列処理、FMAによるfused multiply-add、v1.1.0のバグ修正"

<details>

- **生成時刻**: `2025-12-31T15:18:46Z`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - id: `pending`
    - resource_group: `a-batch-low`
- [ ] **test**
    - performance: `pending`
    - unit: `GFLOPS`
- **params**:
    - simd: `AVX2`
    - fma: `enabled`
    - vector_width: `8 floats (256-bit)`

</details>

---

### v1.1.0
**変更点**: "AVX2 SIMD最適化（初期実装）"
**結果**: バグあり - v1.1.1で修正
**コメント**: "カーネル内のFMA適用に問題があったため修正版を作成"

<details>

- **生成時刻**: `2025-12-31T15:16:00Z`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - status: `skipped`
    - reason: `バグのためv1.1.1に置き換え`

</details>

---

### v1.0.0
**変更点**: "ベースラインコード（sbgemm_nolib.c）の性能測定"
**結果**: 準備中
**コメント**: "BaseCodeのsbgemm.cをそのまま実行してベースライン性能を測定"

<details>

- **生成時刻**: `2025-12-31T15:10:37Z`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - id: `pending`
    - resource_group: `a-batch-low`
- [ ] **test**
    - performance: `pending`
    - unit: `GFLOPS`

</details>
