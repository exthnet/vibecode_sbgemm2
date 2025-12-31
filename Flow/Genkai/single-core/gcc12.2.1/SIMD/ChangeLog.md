# SIMD📁 `ChangeLog.md`
🤖PG1.2
- **ハードウェア**：Genkai single-core（1コア）
- **モジュール**：GCC 12.2.1 + SIMD (AVX2/AVX-512)

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載（PMがテンプレートから生成）

---

### v1.1.0
**変更点**: "B行列事前転置によるSIMD効率向上"
**結果**: 未測定（SSH接続待ち）
**コメント**: "B行列を転置してメモリアクセスを連続化。AVX2内積計算が効率的に動作。BLOCK_K=512に調整"

<details>

- **生成時刻**: `2025-12-31T15:16:46Z`
- [ ] **compile**
    - status: `pending`
    - message: "SSH接続復旧待ち"
- [ ] **job**
    - id: `未投入`
    - resource_group: `a-batch-low`
    - status: `pending`
- [ ] **test**
    - performance: `未測定`
    - unit: `GFLOPS`
- **params**:
    - BLOCK_M: `64`
    - BLOCK_N: `64`
    - BLOCK_K: `512`
    - SIMD: `AVX2`
    - B_transpose: `true`

</details>

---

### v1.0.0
**変更点**: "AVX2ベクトル化とキャッシュブロッキング実装"
**結果**: 未測定（SSH接続待ち）
**コメント**: "BaseCodeにAVX2 intrinsicsとブロッキング(64x64x256)を追加。B行列の列アクセス最適化は次バージョンで対応予定"

<details>

- **生成時刻**: `2025-12-31T15:14:18Z`
- [ ] **compile**
    - status: `pending`
    - message: "SSH接続復旧待ち"
- [ ] **job**
    - id: `未投入`
    - resource_group: `a-batch-low`
    - status: `pending`
- [ ] **test**
    - performance: `未測定`
    - unit: `GFLOPS`
- **params**:
    - BLOCK_M: `64`
    - BLOCK_N: `64`
    - BLOCK_K: `256`
    - SIMD: `AVX2`

</details>

