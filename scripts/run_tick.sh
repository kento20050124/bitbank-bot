#!/bin/zsh
# Run one trading tick. Called by launchd every 30 minutes.
BOT_DIR="/Users/suzukikento/Library/CloudStorage/GoogleDrive-suzukikento@datarein-inc.com/マイドライブ/◾️個人・プレイベート/Claude Code/bitbank-bot"
cd "$BOT_DIR"
"$BOT_DIR/.venv/bin/python" "$BOT_DIR/scripts/tick.py" --paper
