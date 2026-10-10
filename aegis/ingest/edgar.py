from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
import requests
from bs4 import BeautifulSoup

from aegis.config import SETTINGS
from aegis.ingest.base import SourceConnector
from aegis.schemas import TextItem


class SECEdgarConnector(SourceConnector):
    """Official SEC filing connector using keyless data.sec.gov APIs and filing documents."""
    TICKERS = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self) -> None:
        self.session=requests.Session(); self.session.headers.update({"User-Agent":SETTINGS.user_agent,"Accept-Encoding":"gzip, deflate"})

    def _resolve(self,query:str) -> tuple[int,str,str]:
        r=self.session.get(self.TICKERS,timeout=SETTINGS.request_timeout); r.raise_for_status()
        q=query.strip().lower().replace("$","")
        candidates=[]
        for row in r.json().values():
            ticker=str(row.get("ticker","")).lower(); title=str(row.get("title","")).lower()
            if ticker==q: return int(row["cik_str"]),str(row["ticker"]),str(row["title"])
            if q in title: candidates.append(row)
        if not candidates: raise ValueError(f"SEC company not found for query: {query}")
        row=candidates[0]; return int(row["cik_str"]),str(row["ticker"]),str(row["title"])

    def fetch(self,query:str,limit:int=10)->list[TextItem]:
        cik,ticker,company=self._resolve(query)
        u=f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        r=self.session.get(u,timeout=SETTINGS.request_timeout); r.raise_for_status(); data=r.json()
        recent=data.get("filings",{}).get("recent",{})
        forms=recent.get("form",[]); accessions=recent.get("accessionNumber",[]); docs=recent.get("primaryDocument",[])
        dates=recent.get("filingDate",[]); items=recent.get("items",[""]*len(forms))
        accepted=recent.get("acceptanceDateTime",[""]*len(forms))
        out=[]
        keep={"8-K","10-Q","10-K","6-K","20-F","40-F"}
        for i,form in enumerate(forms):
            if form not in keep: continue
            acc=accessions[i]; doc=docs[i]; cikplain=str(cik); accplain=acc.replace("-","")
            filing_url=f"https://www.sec.gov/Archives/edgar/data/{cikplain}/{accplain}/{doc}"
            body=""
            try:
                fr=self.session.get(filing_url,timeout=SETTINGS.request_timeout); fr.raise_for_status()
                soup=BeautifulSoup(fr.text,"html.parser")
                for tag in soup(["script","style"]): tag.decompose()
                body=re.sub(r"\s+"," ",soup.get_text(" ",strip=True))[:18000]
            except Exception:
                body=f"{company} filed {form}. Items: {items[i] if i < len(items) else ''}"
            rawtime=accepted[i] if i < len(accepted) else ""
            try:
                dt=datetime.fromisoformat(rawtime.replace("Z","+00:00")) if rawtime else datetime.fromisoformat(dates[i]).replace(tzinfo=timezone.utc)
                if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
            except Exception: dt=datetime.now(timezone.utc)
            title=f"{company} ({ticker}) {form} filing" + (f" — Items {items[i]}" if i<len(items) and items[i] else "")
            out.append(TextItem(source="SEC EDGAR",title=title,text=body,published_at=dt,url=filing_url,source_domain="sec.gov",
                                metadata={"form":form,"accession":acc,"cik":cik,"ticker":ticker},item_id=hashlib.sha1(acc.encode()).hexdigest()))
            if len(out)>=min(limit,20): break
        return out
