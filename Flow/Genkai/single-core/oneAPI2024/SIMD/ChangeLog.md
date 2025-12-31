# SIMD📁 `ChangeLog.md`
🤖PG1.6
- **ハードウェア**：Genkai (玄界) single-core（1コアのみ）
- **モジュール**：Intel oneAPI 2024.1 (icx, AVX-512)

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：なし

---

### v1.0.0
**変更点**: "AVX-512 SIMD最適化版の初期実装"
**結果**: コンパイル・実行待ち
**コメント**: "512ビットFMA命令、ブロッキング(64x64x256)、BF16→FP32変換最適化"

<details>

- **生成時刻**: `2025-12-31T15:15:59Z`
- [ ] **compile**
    - status: `pending`
    - note: SSH接続待ち
- [ ] **job**
    - status: `pending`
- [ ] **test**
    - status: `pending`
- **params**:
    - BLOCK_M: `64`
    - BLOCK_N: `64`
    - BLOCK_K: `256`
    - SIMD_WIDTH: `512bit (16 floats)`

</details>

---

### v0.0.0 (Baseline)
**変更点**: "BaseCode sbgemm.cのベースライン測定"
**結果**: 測定予定
**コメント**: "3重ループによる単純実装、SIMD最適化前の基準値"

<details>

- **生成時刻**: `2025-12-31T15:11:03Z`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - status: `pending`
- [ ] **test**
    - status: `pending`

</details>
