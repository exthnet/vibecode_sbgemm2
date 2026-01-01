# Cache📁 `ChangeLog.md`
🤖PG1.2
- **ハードウェア**：玄界（Genkai）Node Group A（1コア）
- **モジュール**：GCC 13.3.1

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：なし

---

### v1.1.0
**変更点**: "最適化ブロックサイズ（BLOCK_K=1536, BLOCK_N=480）"
**結果**: 実行中 `pending`
**コメント**: "reference.pdf Table 10-12の最適パラメータを適用"

<details>

- **生成時刻**: `2026-01-01T10:00:00Z`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - id: `4610370`
    - resource_group: `a-batch-low`
    - status: `running`
- [ ] **test**
    - performance: `pending`
    - unit: `GFLOPS`
- **params**:
    - BLOCK_K: `1536`
    - BLOCK_N: `480`

</details>

---

### v1.0.1
**変更点**: "小行列サイズでテスト（500-3000）"
**結果**: 最大 `2.65 GFLOPS` (理論性能の0.14%)
**コメント**: "動作確認完了。ナイーブ実装のため性能は低い。SIMD/ループ最適化が必要"

<details>

- **生成時刻**: `2026-01-01T09:54:00Z`
- [x] **compile**
    - status: `success`
- [x] **job**
    - id: `4610365`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:56:00Z`
    - end_time: `2026-01-01T09:57:00Z`
    - runtime_sec: `43`
    - status: `success`
- [x] **test**
    - performance: `2.65`
    - unit: `GFLOPS`
    - efficiency: `0.14%`
    - results: |
        500: 2.65 GFLOPS
        1000: 2.30 GFLOPS
        1500: 2.56 GFLOPS
        2000: 2.59 GFLOPS
        2500: 2.62 GFLOPS
        3000: 2.60 GFLOPS
- **params**:
    - BLOCK_K: `1024`
    - BLOCK_N: `256`

</details>

---

### v1.0.0
**変更点**: "基本キャッシュブロッキング実装（BLOCK_K=1024, BLOCK_N=256）"
**結果**: タイムアウト `timeout`
**コメント**: "10000x10000行列が10分制限内に完了せず"

<details>

- **生成時刻**: `2026-01-01T09:40:00Z`
- [x] **compile**
    - status: `success`
    - log: `コンパイル成功、sbgemm_v1.0.0実行ファイル生成`
- [x] **job**
    - id: `4610345`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T09:42:39Z`
    - end_time: `2026-01-01T09:52:39Z`
    - runtime_sec: `600`
    - status: `timeout`
- [ ] **test**
    - performance: `N/A`
    - unit: `GFLOPS`
    - note: `大行列で時間超過`
- **params**:
    - BLOCK_K: `1024`
    - BLOCK_N: `256`
    - matrix_sizes: `1000-10000`

</details>

