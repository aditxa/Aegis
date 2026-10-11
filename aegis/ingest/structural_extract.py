"""Public-document structural extraction, independent implementation.

Consumes only supplied HTML bytes. Live access is deliberately NOT implemented:
respect robots, provider ToS, authentication and market-data licensing.
"""
from __future__ import annotations
from hashlib import sha256
from bs4 import BeautifulSoup
import re

def parse_document(html:str, source_url:str='', max_chars:int=2_000_000)->dict:
    if len(html)>max_chars:raise ValueError('HTML too large; reject document')
    soup=BeautifulSoup(html,'html.parser')
    for t in soup(['script','style','noscript','iframe','nav','footer']):t.decompose()
    title=soup.title.get_text(' ',strip=True) if soup.title else ''
    tables=[]
    for t in soup.find_all('table')[:20]:
        rows=[]
        for tr in t.find_all('tr')[:500]:
            cells=[re.sub(r'\s+',' ',x.get_text(' ',strip=True))[:500] for x in tr.find_all(['th','td'],recursive=False)]
            if cells:rows.append(cells)
        if rows:tables.append({'rows':rows,'structural_key':sha256(str([len(r) for r in rows[:10]]).encode()).hexdigest()[:16]})
    text=soup.get_text(' ',strip=True)
    text=re.sub(r'\s+',' ',text)[:max_chars]
    keyvalues=[]
    for t in soup.find_all(['p','li','dt'])[:5000]:
        s=t.get_text(' ',strip=True)
        if ':' in s and len(s)<350:
            a,b=s.split(':',1)
            if 0<len(a)<80 and 0<len(b)<260:keyvalues.append({'key':a.strip(),'value':b.strip()})
    return {'source_url':source_url,'title':title,'text':text,'tables':tables,
            'key_values':keyvalues[:150], 'content_hash':sha256(text.encode()).hexdigest(),
            'source_kind':'supplied_html','extracted_at':'caller_supplies_ingest_timestamp'}

def relevant_change(old:dict,new:dict)->dict:
    if not all('content_hash'in r for r in [old,new]):raise ValueError('invalid extraction result')
    old_kv={x['key']:x['value'] for x in old.get('key_values',[])}
    new_kv={x['key']:x['value'] for x in new.get('key_values',[])}
    delta={k:{'before':old_kv.get(k),'after':new_kv.get(k)} for k in sorted(old_kv.keys()|new_kv.keys()) if old_kv.get(k)!=new_kv.get(k)}
    return {'changed':old['content_hash']!=new['content_hash'],
            'key_value_changes':delta,'requires_review':bool(delta),
            'note':'No semantic materiality prediction; all changes require contextual verification'}
