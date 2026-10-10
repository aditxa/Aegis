from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from aegis.schemas import RiskSignal


class SignalStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                timestamp TEXT,
                ticker TEXT,
                event_classification TEXT,
                impact_score REAL,
                sentiment_score REAL,
                alert_level TEXT,
                payload TEXT
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker_time ON signals(ticker, timestamp)")
        self.conn.commit()

    def put(self, signal: RiskSignal) -> None:
        d = signal.to_dict()
        self.conn.execute(
            "INSERT INTO signals(item_id,timestamp,ticker,event_classification,impact_score,sentiment_score,alert_level,payload) VALUES(?,?,?,?,?,?,?,?)",
            (signal.item_id, signal.timestamp, signal.ticker, signal.event_classification,
             signal.impact_score, signal.sentiment_score, signal.alert_level, json.dumps(d)),
        )
        self.conn.commit()

    def latest(self, ticker: str | None = None, limit: int = 100) -> list[dict]:
        if ticker:
            rows = self.conn.execute("SELECT payload FROM signals WHERE ticker=? ORDER BY id DESC LIMIT ?", (ticker.upper(), limit)).fetchall()
        else:
            rows = self.conn.execute("SELECT payload FROM signals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]
