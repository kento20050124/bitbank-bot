# 戦略強化メモ (2026-05-12)

2025-2026 の研究を踏まえて入れた改良の意図と効果想定。

## 1. Choppiness Index レジームフィルタ

- CI(14) > 61.8 ならエントリー停止（レンジ判定）
- 低CI（強トレンド）ほどスコアにボーナス (`chop_bonus` 最大+10)
- 効果: whipsaw損失の削減、ADX20だけでは拾いきれないレンジ局面の除外

## 2. 手数料込み期待値ゲート

```
score-based payoff(R) = (score/100) × scaling_rr_target
cost(R)               = (2×|taker_fee| + slippage_pct) × price / stop_distance
EV(R)                 = payoff - cost
```

- `EV(R) < expected_value_min_r` ならスキップ
- 低ATR銘柄や低スコアシグナルは数学的に手数料負けする → 機械的に排除

## 3. 小資金（5万円規模）向け銘柄優先度

- `min_lot × price >= 5,000円` （≈ BTC/JPY, ETH/JPY）は score≥60 のみ採用
- 低単価アルト（XRP, DOGE, ADA等）を優先 → リスク%=1%(=500円)が最小ロットを上回りやすい

## 4. 30日相関フィルタ

- スコア降順に評価し、既選択銘柄との pearson 相関 > 0.85 のものは除外
- 効果: max_concurrent_positions=3 の分散効果を保護（BTC+ETH同時保有を抑制等）

## 5. Breakeven SL Move（建値SL移動）

- 含み益が ATR × `breakeven_trigger_r` (=1.0) に達した瞬間 SL を建値に移動
- 既存 ChandelierExit の `max(chandelier, stop_price)` ロジックで自然に保持される
- 効果: 1Rで急失速しても損失ゼロ。残りで大きいトレンドを狙える

## 6. Scaling Out / Chandelier の再調整

- `scaling_rr_target`: 3.0 → 2.0 （ただし `scaling_close_ratio`: 0.5 → 0.3）
- `chandelier_multiplier`: 1.5 → 2.5 （ストップを広げてトレンド継続を許容）
- 思想: クリプトのトレンド持続性を活かす。早めに1/3利確→残り2/3をATR×2.5で長く保持

## 7. 確信度ベース動的サイジング

```
risk_pct = lerp(confidence_min_risk_pct, confidence_max_risk_pct,
                (score - confidence_score_min) / (max - min))
```

- score=50で 0.5%、score=90で 1.5%、score<50は base(1.0%) にフォールバック
- 高確信度シグナルでより大きく張る = 期待値の高い場面でリターン最大化

## 元の固定値からの主な変更

| パラメータ | 旧 | 新 | 理由 |
|---|---|---|---|
| chandelier_multiplier | 1.5 | 2.5 | トレンド継続を許容 |
| scaling_rr_target | 3.0 | 2.0 | 早めの部分利確 |
| scaling_close_ratio | 0.5 | 0.3 | 残り保持を厚く |
| risk_per_trade_pct | 1.0 | 1.0(動的0.5-1.5) | 確信度連動 |
| (new) breakeven_trigger_r | - | 1.0 | リスクフリー化 |
| (new) chop_max_threshold | - | 61.8 | レンジ局面排除 |
| (new) expected_value_min_r | - | 0.0 | 手数料負けトレード排除 |

## 検証方法

- run 後 `gh run view <id> --log | grep "score=\|EV=\|BEST:"` で実シグナルの内訳確認
- 1〜2週間動かしてから `data/candles.db` の `trade_log` を集計し旧期間と比較
