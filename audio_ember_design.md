# Audio Liquid Ember — Multimodal LIF Experiment Design

## 目的
LIF gatingが**モダリティ非依存**であることを実証する。
- テキスト(Transformer): ✅ LIF wins, hierarchy
- テキスト(CfC): ✅ LIF wins, hierarchy
- 音声(CfC): ← **このフェーズ**

## 実験: Speech Commands Classification

### データセット
- **Speech Commands v2** (Google)
- 35単語（yes, no, up, down, left, right, ...）+ silence + unknown
- 105,829発話、各≈1秒、16kHz mono
- Train/Val/Test split標準提供
- torchaudio.datasets.SPEECHCOMMANDS で取得可能

### 前処理
```
Raw audio (16kHz, 1s = 16000 samples)
  → Mel spectrogram (80 bins, 25ms window, 10ms hop)
  → [batch, time≈100, 80]
  → Linear projection: 80 → n_embd (128 or 256)
  → [batch, time≈100, n_embd]
```

### アーキテクチャ
```
mel_spectrogram → projection(80→n_embd) → [CfC + LIF] × N_layers → mean_pool → classifier(n_embd→35)
```

### 条件（ablation）
1. **CfC-only (Base)**: LIFなし
2. **CfC+LIF**: 学習可能閾値LIFゲート

### ハイパーパラメータ（初期案）
| Param | Value |
|-------|-------|
| n_embd | 128 |
| cfc_units | 192 |
| n_layer | 4 |
| dropout | 0.1 |
| batch_size | 64 |
| lr | 1e-3 |
| epochs | 30 |
| optimizer | AdamW |

### 計測指標
1. **精度**: Val/Test accuracy (Base vs LIF)
2. **内部組織化**:
   - ニューロン発火率 per layer (fire_rate)
   - ニューロンentropy per layer
   - 浅層→深層の階層パターン
3. **閾値分布**: 層ごとのLIF threshold学習値

### 期待する結果
- Base: entropy=0 (全ニューロン100%発火、テキストCfCと同じ)
- LIF: progressive hierarchy (L0=broad → L3=selective)
- 同じパターンなら → **LIF gatingはbackbone・modality非依存の普遍的組織化原理**

### テキスト実験との比較表（論文用）
| Backbone | Modality | Metric | Base | LIF | Hierarchy |
|----------|----------|--------|------|-----|-----------|
| Transformer | Text | val_loss | 1.4784 | 1.4673 (-0.75%) | ↑ (strong) |
| CfC | Text | val_loss | 1.4813 | 1.4804 (-0.06%) | ↑ (strong) |
| CfC | Audio | accuracy | ? | ? | ? |

### 実装ステップ
1. [ ] Speech Commands v2 ダウンロード
2. [ ] mel spectrogram前処理パイプライン
3. [ ] AudioLiquidEmber モデル実装（liquid_ember.pyベース）
4. [ ] train_audio.py 作成
5. [ ] Base条件トレーニング (seed=42)
6. [ ] LIF条件トレーニング (seed=42)
7. [ ] 精度比較
8. [ ] analyze_audio.py でentropy分析
9. [ ] 3-seed ablation (42, 668, 1337)

### 所要時間見積もり
- M4 Max 48GB で 1条件あたり ≈30-60分（speech commandsは軽い）
- 3-seed × 2条件 = 6回 ≈3-6時間

---

## 将来拡張（Phase 2以降）
- Whisper encoder特徴量をCfCに入力（より高次の表現）
- ReachyMiniマイク実データでの環境音分類
- テキスト+音声の同時入力（真のマルチモーダル）
- 音声言語モデリング（次のaudio token予測）

2026-02-19 — Tsubasa
