# Memory📁 `ChangeLog.md`
🤖PG1.3 (メモリアクセス最適化担当)
- **ハードウェア**：玄界 (Genkai) Node Group A（1コア）
- **モジュール**：GCC 13.3.1
- **理論ピーク性能**：1945.6 GFLOPS（AMX、ベースクロック1.9GHz）

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：なし

---

### v1.1.0
**変更点**: "レジスタブロッキング(4x4) + 小ブロックサイズ"
**結果**: 理論性能の0.37%達成 `7.17 GFLOPS` (v1.0.0比2.3倍)
**コメント**: "4x4レジスタブロッキングで性能2倍向上。まだ低いためさらなる最適化必要"

<details>

- **生成時刻**: `2026-01-01T09:48:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `compile_v1.1.0.log`
- [x] **job**
    - id: `4610357`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:49:39Z`
    - end_time: `2026-01-01T09:51:34Z`
    - runtime_sec: `115`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance_1024: `7.17 GFLOPS`
    - performance_2048: `6.60 GFLOPS`
    - performance_4096: `5.28 GFLOPS`
    - unit: `GFLOPS`
    - verification: `pass`
- **params**:
    - BLOCK_M: `64`
    - BLOCK_N: `256`
    - BLOCK_K: `256`
    - MR: `4`
    - NR: `4`
    - optimization: `register blocking 4x4, smaller cache blocks`

</details>

---

### v1.0.0
**変更点**: "BUFFER_A/Bパッキング + キャッシュブロッキング実装"
**結果**: 理論性能の0.16%達成 `3.09 GFLOPS`
**コメント**: "基本実装完了。検証成功。性能が低いためv1.1.0でループ順序・プリフェッチ最適化予定"

<details>

- **生成時刻**: `2026-01-01T09:40:37Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `compile_v1.0.0.log`
- [x] **job**
    - id: `4610347`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:42:36Z`
    - end_time: `2026-01-01T09:46:03Z`
    - runtime_sec: `207`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance_1024: `3.09 GFLOPS`
    - performance_2048: `3.08 GFLOPS`
    - performance_4096: `3.02 GFLOPS`
    - unit: `GFLOPS`
    - verification: `pass (expected [[58,64],[139,154]])`
- **params**:
    - BLOCK_M: `96`
    - BLOCK_N: `480`
    - BLOCK_K: `1536`
    - optimization: `BUFFER_A/B packing, cache blocking, loop unrolling(4)`

</details>
