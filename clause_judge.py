import os
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
import re
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import base64
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()


def get_text(response) -> str:
    """Safely pull text from a model response, whether .content is a string or a list."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Gemini sometimes returns a list of parts; join their text
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()

class Clause(BaseModel):
    """One segmented clause in a legal document."""
    clause_id: str
    text: str
    category: str = Field(
        description="Clause category: payment, termination, indemnity, "
                    "confidentiality, dispute_resolution, liability, penalty, other"
    )


class SegmentationResult(BaseModel):
    """What the segmentation node returns: contract type + all clauses."""
    contract_type_guess: str = Field(
        description="e.g. 'rental agreement', 'employment contract', 'NDA', "
                    "'loan agreement', 'service contract'"
    )
    clauses: list[Clause]


class StatuteFinding(BaseModel):
    """One clause's compliance check against Indian statute."""
    clause_id: str
    compliant: bool
    statute_reference: Optional[str] = Field(
        default=None,
        description="e.g. 'Indian Contract Act 1872, Sec 28'"
    )
    issue: Optional[str] = Field(
        default=None,
        description="Plain-language explanation of the concern, if any"
    )


class StatuteReview(BaseModel):
    """The full output of the statute worker."""
    findings: list[StatuteFinding]


class CaseLawFinding(BaseModel):
    """One clause's relevant case-law precedent, if any."""
    clause_id: str
    relevant_case: Optional[str] = None
    summary: Optional[str] = Field(
        default=None,
        description="1-2 sentences on why this precedent matters"
    )


class CaseLawReview(BaseModel):
    """The full output of the case-law worker."""
    findings: list[CaseLawFinding]


class RiskFinding(BaseModel):
    """One clause's risk assessment."""
    clause_id: str
    risk_score: int = Field(ge=1, le=5, description="1=low risk, 5=high risk")
    one_sided: bool = Field(description="True if it disproportionately favors one party")
    note: str = Field(description="Short explanation of the score")


class RiskReview(BaseModel):
    """The full output of the risk worker."""
    findings: list[RiskFinding]
    missing_protections: list[str] = Field(
        default_factory=list,
        description="Standard clauses the contract seems to lack entirely"
    )


## ---------------------------------------------------------------------------
## Graph state — the shared object every node reads from and writes to
## ---------------------------------------------------------------------------

class State(TypedDict, total=False):
    upload_bytes: bytes| str            # ← add: raw uploaded file
    upload_type: str               # ← add: "image" or "pdf"
    contract_text: str             # input: raw contract
    contract_type: str             # set by segmentation
    clauses: list[dict]            # set by segmentation
    statute_findings: list[dict]   # set by statute_worker
    case_law_findings: list[dict]  # set by case_law_worker
    risk_findings: list[dict]      # set by risk_worker
    missing_protections: list[str] # set by risk_worker
    report_markdown: str           # set by aggregator (final output)

## ---------------------------------------------------------------------------
## The AI model
## ---------------------------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=2,
)


## ---------------------------------------------------------------------------
## Node 1: read the contract, split it into categorized clauses
## ---------------------------------------------------------------------------
def ingest_and_segment(state: State, config: RunnableConfig):
    """Classify the contract type and break it into categorized clauses."""
    prompt = f"""You are a contract analyst. Read the contract below and:
1. Guess the overall contract type (e.g. rental agreement, NDA, employment contract).
2. Split it into individual clauses.
3. Give each clause a short id (C1, C2, C3, ...) and a category from this set:
   payment, termination, indemnity, confidentiality, dispute_resolution,
   liability, penalty, other.

Contract:
---
{state['contract_text']}
---"""
    structured_model=model.with_structured_output(SegmentationResult)
    result=structured_model.invoke(prompt)
     # Flatten the Pydantic Clause objects into plain dicts for graph state
    clauses = []
    for c in result.clauses:
        clauses.append({
            "clause_id": c.clause_id or str(uuid.uuid4())[:8],   # ← "clause_id", no s
            "text": c.text,
            "category": c.category
        })
    return{
        "contract_type":result.contract_type_guess,
        "clauses":clauses
    }

STATUTE_REFERENCE = """
Indian Contract Act, 1872:
- Sec 10: Agreements are contracts if made by free consent of competent parties, for lawful consideration and object.
- Sec 23: Consideration or object is unlawful if forbidden by law, fraudulent, or against public policy.
- Sec 27: An agreement restraining anyone from exercising a lawful profession or trade is void (narrow exceptions only).
- Sec 28: An agreement that absolutely restricts a party from enforcing their rights through ordinary legal proceedings is void. (Relevant to clauses that try to bar a party from approaching any court.)
- Sec 73/74: Compensation for breach; a penalty clause is only enforceable up to a genuine pre-estimate of loss, not as a punitive amount.

Consumer Protection Act, 2019:
- Unfair contract terms (e.g. one-sided termination rights, or excessive non-refundable deposits) may be challengeable.

Indian Stamp Act, 1899 / State Stamp Acts:
- Some contracts (esp. rental/lease) require stamp duty; an unstamped document may be inadmissible as evidence.

Information Technology Act, 2000:
- Sec 10A: Electronic contracts and e-signatures are valid and enforceable.
"""

def statute_worker(state:State,config:RunnableConfig):
    """check each clause against Indian statute for compilance concerns."""
    clauses=state["clauses"]
    # Build a readable block listing every clause with its id and category
    clause_block="\n\n".join(
        f"[{c['clause_id']}] ({c['category']}) {c['text']}" for c in clauses
    )

    prompt = f"""You are an Indian contract-law compliance checker. Using the
statute reference below, assess each clause for compliance concerns under
Indian law. If a clause is fine, mark compliant=true with no issue. If a
clause is problematic, mark compliant=false, explain the issue in plain
language, and cite the relevant statute section.

<statute_reference>
{STATUTE_REFERENCE}
</statute_reference>

<clauses>
{clause_block}
</clauses>

Return exactly one finding per clause, matching each clause's id."""
    structured_model=model.with_structured_output(StatuteReview)
    result=structured_model.invoke(prompt)
    return{"statute_findings":[f.model_dump()for f in result.findings]}


def risk_worker(state:State,config:RunnableConfig):
    """Score each clause's risk 1-5, flag one-sided terms, and list missing protections."""
    clauses=state["clauses"]
    clause_block="\n\n".join(
        f"[{c['clause_id']}] ({c['category']}) {c['text']}" for c in clauses
    )

    categories_present = sorted({c["category"] for c in clauses})

    prompt = f"""You are a contract risk analyst. Assess this contract from the
point of view of the weaker party signing it (assume they are the tenant,
employee, borrower, or customer — not the party who drafted it).

For each clause:
- Give a risk_score from 1 (low risk) to 5 (high risk).
- Set one_sided=true if the clause disproportionately favors the other party.
- Write a short note explaining the score.

Also identify standard protections this contract appears to be MISSING entirely.
The clause categories present are: {categories_present}. Common protections to
check for: dispute resolution mechanism, termination notice period, liability
cap, confidentiality, refund/deposit-return terms.

<clauses>
{clause_block}
</clauses>

Return one finding per clause id, plus a list of missing protections."""
    structured_model=model.with_structured_output(RiskReview)
    result=structured_model.invoke(prompt)

    return {
        "risk_findings":[f.model_dump()for f in result.findings],
        "missing_protections":result.missing_protections
    }




## ---------------------------------------------------------------------------
## Indian Kanoon API
## ---------------------------------------------------------------------------

INDIAN_KANOON_API_KEY = os.environ.get("INDIAN_KANOON_API_KEY", "")
INDIAN_KANOON_SEARCH_URL = "https://api.indiankanoon.org/search/"

# Only these clause categories are worth a paid case-law lookup.
# A plain "payment" clause has no interesting precedent, so we skip it.

CASE_LAW_WORTHY_CATEGORIES = {
    "termination", "indemnity", "dispute_resolution",
    "liability", "confidentiality", "penalty",
}

def search_indian_kanoon(query: str, max_results: int = 3)-> list[dict]:
    """Search Indian Kanoon. Returns [] on any failure (never crashes the graph)."""
    if not INDIAN_KANOON_API_KEY:
        return []
    try:
        resp=requests.post(
            INDIAN_KANOON_SEARCH_URL,
            headers={"Authorization": f"Token {INDIAN_KANOON_API_KEY}"},
            data={"formInput": query, "pagenum": 0},   # note: formInput, page starts at 0
            timeout=15,
        )
        if resp.status_code != 200:
            print("Indian Kanoon API error:", resp.status_code, resp.text[:500])
            return []
        docs= resp.json().get("docs", [])[:max_results]
        return [
            {
                "title": re.sub("<[^<]+?>", "", d.get("title", "")),  # strip HTML tags
                "docid": d.get("tid", ""),
                "snippet": re.sub("<[^<]+?>", "", d.get("headline", ""))[:300],  # strip HTML tags
            }for d in docs
        ]
    except Exception:
        return []

def case_law_worker(state: State, config: RunnableConfig):
    """For risky clauses, search Indian Kanoon and pick the most relevant precedent."""
    clauses = state["clauses"]
    contract_type = state.get("contract_type", "contract")
    worthy = [c for c in clauses if c["category"] in CASE_LAW_WORTHY_CATEGORIES]

    findings = []
    for c in worthy:
        # Step 1: ask the model to turn the clause into a focused legal search query
        query_prompt = f"""What is the core legal issue in this contract clause?
Reply with a short Indian-law search query (statute sections, legal doctrine,
key terms) — no more than 12 words, no explanation.

Clause: {c['text']}"""
        try:
            search_query = get_text(model.invoke(query_prompt))
        except Exception:
            search_query = f"{c['text']} India contract law"

        # Step 2: search Indian Kanoon with that focused query
        hits = search_indian_kanoon(search_query)
        if not hits:
            continue

        hits_block = "\n".join(f"- {h['title']}: {h['snippet']}" for h in hits)

        # Step 3: ask the model which case actually matters
        prompt = f"""Given this contract clause and these Indian Kanoon search
results, pick the single most relevant case (if any) and explain in 1-2
sentences why it matters for this clause. If none are genuinely relevant,
leave relevant_case empty.

Clause ({c['category']}): {c['text']}

Search results:
{hits_block}"""

        structured_model = model.with_structured_output(CaseLawFinding)
        try:
            result = structured_model.invoke(prompt)
            result.clause_id = c["clause_id"]
            findings.append(result.model_dump())
        except Exception:
            continue

    return {"case_law_findings": findings}


def aggregator(state: State):
    """Merge all worker outputs into a single markdown report. No AI/API — pure assembly."""

    # Turn each list into a lookup dict keyed by clause_id, so we can find
    # "the risk finding for C3" instantly instead of searching the whole list.
    clauses = {c["clause_id"]: c for c in state.get("clauses", [])}
    statute = {f["clause_id"]: f for f in state.get("statute_findings", [])}
    risk = {f["clause_id"]: f for f in state.get("risk_findings", [])}

    # A clause could have more than one case-law finding, so group into lists.
    case_law = {}
    for f in state.get("case_law_findings", []):
        case_law.setdefault(f["clause_id"], []).append(f)

    lines = [f"# Contract Review — {state.get('contract_type', 'Unknown type')}", ""]

    # ---- Summary section (counts at the top) ----
    high_risk = [r for r in state.get("risk_findings", []) if r["risk_score"] >= 4]
    concerns = [s for s in state.get("statute_findings", []) if not s["compliant"]]

    lines.append("## Summary")
    lines.append(f"- **{len(clauses)}** clauses reviewed")
    lines.append(f"- **{len(high_risk)}** high-risk clauses (score ≥ 4)")
    lines.append(f"- **{len(concerns)}** potential compliance concerns")
    if state.get("missing_protections"):
        lines.append(f"- **Missing protections:** {', '.join(state['missing_protections'])}")
    lines.append("")

    # ---- Clause-by-clause detail ----
    lines.append("## Clause-by-clause findings")
    for cid, clause in clauses.items():
        lines.append(f"### [{cid}] {clause['category'].replace('_', ' ').title()}")
        preview = clause["text"][:200] + ("..." if len(clause["text"]) > 200 else "")
        lines.append(f"> {preview}")
        lines.append("")

        r = risk.get(cid)
        if r:
            flag = " ⚠️ **one-sided**" if r["one_sided"] else ""
            lines.append(f"- **Risk:** {r['risk_score']}/5{flag} — {r['note']}")

        s = statute.get(cid)
        if s:
            if s["compliant"]:
                lines.append("- **Compliance:** OK")
            else:
                lines.append(f"- **Compliance concern:** {s.get('issue', '')} "
                             f"({s.get('statute_reference', 'see reference')})")

        for cl in case_law.get(cid, []):
            if cl.get("relevant_case"):
                lines.append(f"- **Precedent:** {cl['relevant_case']} — {cl.get('summary', '')}")

        lines.append("")

    report = "\n".join(lines)
    return {"report_markdown": report}

# ---- Orchestrator: a pass-through node that exists as the fan-out point ----
def orchestrator(state: State):
    """Does nothing itself — it's just where the parallel fan-out happens."""
    return {}

# ---- The fan-out: send the clauses to all three analysis workers at once ----
def route_to_workers(state: State):
    return [
        Send("statute_worker", state),
        Send("risk_worker", state),
        Send("case_law_worker", state),
    ]

def extract_text_from_upload(state: State, config: RunnableConfig):
    """Turn an uploaded image or PDF into plain contract text.

    Expects state to contain either:
      - 'contract_text' (already plain text — pass straight through), or
      - 'upload_bytes' + 'upload_type' ('image' or 'pdf')
    """

    # If plain text was passed directly, nothing to extract.
    if state.get("contract_text"):
        return {"contract_text": state["contract_text"]}

    upload_bytes = state.get("upload_bytes")
    upload_type = state.get("upload_type", "image")

    if not upload_bytes:
        return {"contract_text": ""}

    if isinstance(upload_bytes, str):
        upload_bytes = base64.b64decode(upload_bytes)
    # ---- PDF: try direct text extraction first (digital PDFs) ----
    if upload_type == "pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(upload_bytes))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            # If we got real text, it's a digital PDF — use it.
            if len(text.strip()) > 100:
                return {"contract_text": text}
            # Otherwise it's a scanned PDF (images) — fall through to vision.
        except Exception:
            pass  # fall through to vision

    # ---- Image (or scanned PDF): use Gemini vision ----
    b64 = base64.b64encode(upload_bytes).decode("utf-8")
    mime = "application/pdf" if upload_type == "pdf" else "image/jpeg"

    message = HumanMessage(content=[
        {"type": "text", "text": "Transcribe all text in this contract exactly as written. Output only the contract text, no commentary."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])

    response = model.invoke([message])
    text = get_text(response)   # ← reuses your helper to handle the list format

    return {"contract_text": text}


# ---- Build the graph ----
builder = StateGraph(State)
builder.add_node("extract_text_from_upload", extract_text_from_upload)
builder.add_node("ingest_and_segment", ingest_and_segment)
builder.add_node("orchestrator", orchestrator)
builder.add_node("statute_worker", statute_worker)
builder.add_node("risk_worker", risk_worker)
builder.add_node("case_law_worker", case_law_worker)
builder.add_node("aggregator", aggregator)

# The flow:
builder.add_edge(START, "extract_text_from_upload")
builder.add_edge("extract_text_from_upload", "ingest_and_segment")
builder.add_edge("ingest_and_segment", "orchestrator")  # segment → fan-out point

# orchestrator fans out to all three workers in parallel
builder.add_conditional_edges(
    "orchestrator",
    route_to_workers,
    ["statute_worker", "risk_worker", "case_law_worker"],
)

# all three workers feed into the aggregator

builder.add_edge("statute_worker", "aggregator")
builder.add_edge("risk_worker", "aggregator")
builder.add_edge("case_law_worker", "aggregator")

builder.add_edge("aggregator", END)                 # aggregator → done

graph = builder.compile()

if __name__ == "__main__":
    with open("sample_contract.jpg", "rb") as f:
        image_bytes = f.read()

    # Full pipeline from an UPLOADED IMAGE, one call
    result = graph.invoke({
        "upload_bytes": image_bytes,
        "upload_type": "image",
    })
    print(result["report_markdown"])