# SIMD📁 `ChangeLog.md`
🤖PG1.1
- **ハードウェア**：玄界 (Genkai) a-batch-low （1コア）
- **モジュール**：GCC 8.5.0 (default) + SIMD (AVX2/AVX-512)

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：なし

---

### v1.1.0
**変更点**: "キャッシュブロッキング最適化の追加"
**結果**: ローカル作成完了 `pending_test`
**コメント**: "BLOCK_M=64, BLOCK_N=64, BLOCK_K=256のタイルベース計算。L1/L2キャッシュ効率を改善"

<details>

- **生成時刻**: `2025-12-31T15:30:00Z`
- [ ] **compile**
    - status: `pending`
    - message: "SSH復旧待ち"
- [ ] **job**
    - status: `pending`
- [ ] **test**
    - status: `pending`
- **params**:
    - nodes: `1`
    - strategy: `AVX2 SIMD + Cache Blocking`
    - block_m: `64`
    - block_n: `64`
    - block_k: `256`

</details>

---

### v1.0.0
**変更点**: "AVX2 SIMDベクトル化の基本実装"
**結果**: ローカル作成完了 `pending_test`
**コメント**: "bf16→fp32変換をAVX2で8要素同時処理。FMA命令使用"

<details>

- **生成時刻**: `2025-12-31T15:25:00Z`
- [ ] **compile**
    - status: `pending`
    - message: "SSH復旧待ち"
- [ ] **job**
    - status: `pending`
- [ ] **test**
    - status: `pending`
- **params**:
    - nodes: `1`
    - strategy: `AVX2 SIMD (8-wide vectorization)`
    - features: `bf16x8_to_fp32, FMA`

</details>

---

### v0.0.1
**変更点**: "初期化 - SSH接続問題の調査"
**結果**: SSH接続がパスフレーズ入力待ちで保留 `pending`
**コメント**: "Desktop Commander MCPでのSSH接続確立を試行。パスフレーズ入力が必要な可能性あり"

<details>

- **生成時刻**: `2025-12-31T15:20:00Z`
- [ ] **compile**
    - status: `not_started`
    - message: "SSH接続問題のため未実行"
- [ ] **job**
    - status: `not_started`
- [ ] **test**
    - status: `not_started`
- **params**:
    - nodes: `1`
    - strategy: `SIMD (AVX2/AVX-512)`

</details>

---

## 次の計画

### v1.2.0 (計画)
- AVX-512への拡張（サポートされている場合）
- プリフェッチ命令の導入
- より大きなブロックサイズの検討
