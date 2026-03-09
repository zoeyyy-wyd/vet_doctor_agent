 # agent/triage.py
import json, re
from typing import Dict, Any, Tuple, Optional

TRIAGE_SCHEMA = """
Return JSON ONLY:
{
 "triage": "green"|"yellow"|"red",
 "final_summary": string,
 "key_risks": [string],
 "what_to_monitor": [string],
 "when_to_seek_vet": [string],
 "vet_visit_checklist": [string],
 "citations": [string],
 "disclaimer": string
}
Rules:
- Use ONLY CASE + EVIDENCE; if unsure, say unsure.
- No definitive diagnosis.
- No medications, dosing, or treatment plan.
- If severe breathing difficulty/collapse/seizure/uncontrolled bleeding are present or strongly suggested -> triage="red".
"""

REQUIRED_KEYS = {
    "triage","final_summary","key_risks","what_to_monitor",
    "when_to_seek_vet","vet_visit_checklist","citations","disclaimer"
}

BANNED_PATTERNS = [
    r"\bmg\b", r"\bml\b", r"\btablet\b", r"\bcapsule\b",
    r"\badminister dose\b", r"\bibuprofen\b", r"\bacetaminophen\b",
    r"\bdiagnosed with\b", r"\bdefinitely has\b"
]

def validate_triage(obj: Dict[str, Any]) -> Tuple[bool, str]:
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"missing keys: {missing}"
    if obj["triage"] not in ("green","yellow","red"):
        return False, "triage must be green/yellow/red"
    return True, "ok"

def safety_check(text: str):
    hits = []
    for p in BANNED_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits

def build_query(case_summary: Dict[str, Any]) -> str:
    signs = ", ".join(case_summary.get("suspected_abnormal_signs", [])[:8])
    red = ", ".join(case_summary.get("notable_red_flags_seen", [])[:5])
    return f"abnormal signs: {signs}; red flags: {red}".strip()

def groq_triage(case_summary: Dict[str, Any], audio_text: str, retriever, groq_client,
               model="llama-3.3-70b-versatile", k=4):
    query = build_query(case_summary)
    hits = retriever.invoke(query)

    evidence_blocks, citations = [], []
    for i, h in enumerate(hits):
        src = h.metadata.get("source", f"source_{i}")
        citations.append(str(src))
        evidence_blocks.append(f"[Source: {src}]\n{h.page_content}")

    evidence = "\n\n".join(evidence_blocks)

    prompt = f"""You are a cautious veterinary assistant.

CASE (from video frames + audio transcript):
{json.dumps(case_summary, ensure_ascii=False, indent=2)}

Audio transcript (may be empty):
{(audio_text[:1200] if audio_text else "(none)")}

EVIDENCE (retrieved):
{evidence}

{TRIAGE_SCHEMA}
"""

    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()

    # loose parse: try to find {...}
    s, e = raw.find("{"), raw.rfind("}")
    triage = json.loads(raw[s:e+1]) if (s!=-1 and e!=-1 and e>s) else json.loads(raw)

    # ensure citations present
    if not triage.get("citations"):
        triage["citations"] = list(dict.fromkeys(citations))

    debug = {"query": query, "sources": citations, "raw": raw, "evidence": evidence}
    return triage, debug

def repair_with_groq(bad_output: str, case_summary: Dict[str, Any], audio_text: str,
                     groq_client, evidence: Optional[str] = None,
                     model="llama-3.3-70b-versatile"):
    fix_prompt = f"""Fix the JSON to match the schema EXACTLY. Return JSON ONLY.
No diagnosis. No medication names/dosing.
Use ONLY CASE + EVIDENCE.

CASE:
{json.dumps(case_summary, ensure_ascii=False, indent=2)}

AUDIO:
{(audio_text[:800] if audio_text else "(none)")}

EVIDENCE:
{(evidence[:3000] if evidence else "(none)")}

BAD_OUTPUT:
{bad_output}

Schema keys:
{sorted(list(REQUIRED_KEYS))}
triage must be one of green/yellow/red.
"""
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":fix_prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    s, e = raw.find("{"), raw.rfind("}")
    return json.loads(raw[s:e+1]) if (s!=-1 and e!=-1 and e>s) else json.loads(raw)

def safe_triage(case_summary: Dict[str, Any], audio_text: str, retriever, groq_client,
               max_repairs=1, verbose=True):
    triage_obj, debug = groq_triage(case_summary, audio_text, retriever, groq_client)
    rep_ok, rep_msg = validate_triage(triage_obj)
    banned = safety_check(json.dumps(triage_obj, ensure_ascii=False))

    repairs = 0
    while repairs < max_repairs and ((not rep_ok) or banned):
        repairs += 1
        if verbose:
            print(f"Repairing output (attempt {repairs}/{max_repairs})...")
        triage_obj = repair_with_groq(
            bad_output=debug["raw"],
            case_summary=case_summary,
            audio_text=audio_text,
            groq_client=groq_client,
            evidence=debug.get("evidence")
        )
        rep_ok, rep_msg = validate_triage(triage_obj)
        banned = safety_check(json.dumps(triage_obj, ensure_ascii=False))

    if (not rep_ok) or banned:
        raise RuntimeError(f"Unsafe triage output. schema_ok={rep_ok} msg={rep_msg} banned={banned}")

    return triage_obj, debug