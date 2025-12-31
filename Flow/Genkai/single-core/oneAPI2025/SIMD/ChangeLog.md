# SIMD📁 `ChangeLog.md`
🤖PG1.8
- **ハードウェア**：Genkai (玄界) a-batch-low （1コア）
- **モジュール**：Intel oneAPI 2025.1.3 (icx)

## 目標
- BF16行列積和演算（SBGEMM）のSIMD最適化
- 参照論文: reference.pdf - Intel AMXを使用したGEMM実装最適化
- 理論性能目標: 環境確認後に設定（AMD EPYC環境ではAVX2/AVX-512を使用）

## Change Log

- 基本の型：`ChangeLog_format.md`に記載
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載（PMがテンプレートから生成）

---

### v0.0.1
**変更点**: "ベースコード確認・環境調査準備"
**結果**: SSH接続失敗 `No route to host`
**コメント**: "ネットワーク環境の問題でスパコン接続不可。ローカルでコード準備を進行"

<details>

- **生成時刻**: `2025-12-31T15:11:54Z`
- [ ] **compile**
    - status: `pending`
    - message: "SSH接続待ち"
- [ ] **job**
    - id: `-`
    - resource_group: `a-batch-low`
    - status: `pending`
- [ ] **test**
    - status: `pending`
- **params**:
    - nodes: `1`
    - cores: `1`

</details>

---

## 最適化戦略

### Phase 1: ベースライン確立
1. BaseCodeのsbgemm.cをコンパイル・動作確認
2. 性能測定用ベンチマークコード作成
3. 初期性能測定

### Phase 2: SIMD最適化
1. **AVX-512 BF16命令の活用**（Intel oneAPI 2025）
   - `_mm512_dpbf16_ps`: BF16ドット積命令
   - 16要素のBF16ペアを並列処理
2. **ループアンローリング**
3. **データアライメント最適化**
4. **キャッシュブロッキング**

### 参考: reference.pdf要点
- Intel AMX: Tile Matrix Multiply (TMUL)を使用
- 理論性能: 1945.6 GFLOPS/core (Sapphire Rapids, 1.9GHz)
- Tiling_B方式で65%効率達成
- キャッシュブロッキング: k=1536, n=480が最適
