# DB永続化と collector の挙動修正 (2026-05-12)

## 背景

2026-04-27 を最後にBOTが新規エントリーを一切しなくなった。GitHub Actions
の `gh run view` ログでは全銘柄が `No more candle data available / 0 candles
saved` を返し、`_scan` は `len(df) < 200` で即 None。

## 根本原因

### 1. DB永続化が機能していなかった

`.github/workflows/trade.yml` は毎回 `secrets.DB_B64` (静的Base64) からDBを
復元するだけで、run 間でデータを引き継いでいなかった。`upload-artifact`
はしていたが download していない。`Save database to cache` ステップは名前と
裏腹に `actions/cache` を使わず `wc -c` を echo するだけ。

結果として、CI環境のDBは「DB_B64生成時のスナップショット」で固定され、
過去数か月分の最新ローソク足が一切蓄積されていなかった。

### 2. ccxt.bitbank.fetch_ohlcv の日別エンドポイント特性

bitbank公開APIの `/v1/{pair}/candlestick/{tf}/{YYYYMMDD}` は日別の
エンドポイント。ccxtは `since` ms を渡されるとその日付に変換して
1日分のキャンドルだけを返す。当該日にデータが無い（または未公開）
場合、空配列を返す。

旧 collector はこれを「もう取れるものがない」と解釈して即 break。
よって新規取得が常にゼロ件で終わっていた。

## 修正内容

### `actions/cache` でDBを run 間永続化 (`.github/workflows/trade.yml`)

```yaml
- name: Restore DB cache
  uses: actions/cache/restore@v4
  with:
    path: data/candles.db
    key: bitbank-candles-db-${{ github.run_id }}
    restore-keys: |
      bitbank-candles-db-

- name: Save DB cache
  if: always()
  uses: actions/cache/save@v4
  with:
    path: data/candles.db
    key: bitbank-candles-db-${{ github.run_id }}
```

- 毎 run ユニークキー (run_id) で保存し、restore_keys プレフィックスで前回の最新を取得
- DB_B64 secret は cache miss 時のフォールバックとして残してある
- `concurrency: bitbank-trade / cancel-in-progress: false` で同一時刻の重複起動を防止（DB競合回避）

### collector.py: 空応答時に1日進めて続行

`collect_historical_candles` / `fetch_latest_candles` 両方:

- 空配列が来たら `since_ms += 86400_000` で次の日にスキップ
- 連続 N 日空 → break（収集はあきらめる）
- `fetch_latest_candles` は `since=now-2days` で日跨ぎを取りこぼさない
- `INSERT OR REPLACE` なので重複保存しても問題なし

## 検出のポイント

- `gh run view <id> --log | grep "0 candles saved"` が全銘柄で並ぶ
- ローカル `data/candles.db` の `MAX(timestamp)` が「BOTが取引を停止した日」と一致
- `secrets.DB_B64` のサイズ確認: 数 KB しかなければほぼ空

## 関連: 既知の不整合データ

`positions` テーブルに `id=3 XLM/JPY current_amount=224.4152
realized_pnl=-6311.0 state=closed` という不整合行が残存。
trade_log に対応する exit が無いため過去の手動クリーンアップの痕跡と思われる。
PnL集計には影響するが、CI環境（cache 経由のDB）には引き継がれないため
ライブ取引には影響しない。手動で `UPDATE positions SET current_amount=0
WHERE id=3` で正規化可能。
