#!/usr/bin/env python3
"""
survival.py — Survival Scalar Engine for LLM Output Quality  [v2.1 — v4 PROMOTED]

S(x) = kappa(x) / (kappa(x) + lambda1 * delta_L(x) + lambda2 * delta_G(x) + eps_u)

π_E (v4 — PRIMARY OUTPUT POLICY):
    kappa_v4(x)   — mean similarity of baseline vs each context response
    delta_L_v4(x) — variance of [1.0, sim(baseline, ctx_i)] (anchored)
    delta_G(x)    — global inconsistency (shared)
    All production decisions go through v4. No fallback to v1.

π_S (v1 — FROZEN SHADOW VALIDATOR / AUDIT LAYER):
    kappa_v1(x)   — pairwise cosine similarity of (baseline + perturbed) responses
    delta_L_v1(x) — variance of pairwise similarities
    Runs in parallel as comparator only. Never makes production decisions.
    Logged for weekly audit and failure-mode analysis.

DECISION GATE (v4 only):
    accept    if S > tau_h (0.70)
    review    if tau_l < S <= tau_h
    reject    if S <= tau_l (0.20)

SAFETY RULE: if v4 and v1 disagree on high-impact outputs →
    route to shadow review (π_S audit), not automatic acceptance.

CALIBRATED PARAMETERS (locked — v2.1):
    lambda1 = 0.5, lambda2 = 0.5
    tau_h = 0.70, tau_l = 0.20
    Promoted: 652 samples, ba(v4)=1≤ba(v1)=1, gr(v4)=0≤gr(v1)=1

Usage:
    from survival import SurvivalEngine, SurvivalConfig

    cfg = SurvivalConfig(deepseek_api_key="sk-...")
    engine = SurvivalEngine(cfg)

    # v4 primary (production output)
    result = engine.evaluate("What is the speed of light?")
    print(f"S={result.S:.3f} decision={result.decision}")

    # Shadow audit mode (v4 decides, v1 validates)
    result = engine.evaluate_shadow("Explain quantum computing")
    print(f"v4={result['v4']['decision']} v1={result['v1']['decision']}")
    if result.get('needs_shadow_review'):
        print(f"HIGH-IMPACT DIVERGENCE → flagged for audit")
"""

import json
import math
import os
import sys
import time
import hashlib
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PROMPTS_PATH = os.path.join(DIR, "data", "raw_prompts.jsonl")
DISAGREEMENT_LOG_PATH = os.path.join(DIR, "logs", "disagreement_cases.jsonl")

# ── Monitor action hook (set by batch runner before eval, routing-layer only) ──
_current_monitor_action = "none"

def set_monitor_action(action: str) -> None:
    """Set the current monitor action for the next evaluate_shadow() call.
    Used by the batch runner to signal routing interventions.
    Does NOT affect scoring, gating, or model behavior.
    Values: 'none' | 'forced_review' | 'tightened_threshold'
    """
    global _current_monitor_action
    _current_monitor_action = action

# High-impact decision zones: disagreements here require shadow review
# v4 says "accept" but S is in the danger zone near tau_h, or
# v4 says "reject" but S is close to tau_l (potential over-rejection)
HIGH_IMPACT_ZONE = {
    "accept_near_threshold": True,   # v4=accept with S in [tau_h, tau_h+0.10]
    "reject_near_threshold": True,   # v4=reject with S in [tau_l-0.10, tau_l]
    "cross_tier": True,              # v4 and v1 differ by >1 tier (accept vs reject)
}

# Promotion decision rule (validated, locked):
# Promote v4 as π_E if ALL of:
#   good_rejected(v4) ≤ good_rejected(v1)
#   bad_accepted(v4) ≤ bad_accepted(v1)
#   divergence(v4) ≤ divergence(v1) + ε  (ε = 0.02)
#   evaluated over ≥ 500 samples
# STATUS: PROMOTED (652 samples, all conditions met)


# ═══════════════════════════════════════════════════════════════
# FAILURE MODE CLASSIFIER (lightweight, inline — observability only)
# ═══════════════════════════════════════════════════════════════

def _classify_failure_mode(prompt: str) -> str:
    """Classify a prompt into a failure mode cluster.
    Used ONLY for observability logging. Does NOT affect scoring or decisions.
    Returns the cluster label string."""
    import re
    p = prompt.lower().strip()

    # Trick/gotcha
    if re.search(r"bury.*survivor|plane.*crash.*border|word.*word.*word"
                 r"|compile.*breakfast|colorless green ideas", p):
        return "trick_question"
    # Impossible
    if re.search(r"infinite|zero latency|solve everything|complete guide to life"
                 r"|teach me everything|impossible|perpetual", p):
        return "impossible_request"
    # Scope overreach
    if re.search(r"teach me everything|complete guide|explain everything"
                 r"|tell me everything|write a book|what do i need to know about"
                 r"|all about|tell me what i should know", p):
        return "scope_overreach"
    # Debug underspecified
    if re.search(r"fix my code|it'?s broken|doesn'?t work|why doesn'?t it"
                 r"|help.*it'?s broken|500 error|error.*what('?s)? the problem"
                 r"|undefined is not a function|race condition|make (it|this) (fast|bigger)"
                 r"|show me how it works", p):
        return "debug_underspecified"
    # Opinion/debate
    if re.search(r"(worth|should i|better|which is|your take|what do you think|opinion)"
                 r"|(some (say|developers|people)|debate|controversial|argue)"
                 r"|(typescript.*worth|framework.*should|sql.*programming language)", p):
        return "opinion_debate"
    # Confused user
    if re.search(r"the thing|won'?t let me|my phone|my computer"
                 r"|my daughter says|my son says|i need the one"
                 r"|doing something weird|letters.*bigger|accept cookie"
                 r"|first (smartphone|computer|time)|retiring"
                 r"|password manager|2-factor|two.factor|2fa"
                 r"|why does (everything|technology|website).*"
                 r"|(bank|email).*link|signal", p):
        return "confused_user"

    # Below here: needs S-score context, but we classify on prompt text alone.
    # For disagreement logging, the domain_knowledge tag will be applied
    # in log_disagreement where we have S scores.

    # Vague ambiguous (short + vague start)
    vague_start = re.search(
        r"^(can you )?(help|explain more|just tell me|fix this|improve this|optimize this|debug|show me)"
        r"|^(what should i|tell me what|what do i|how do i( make| get| handle))"
        r"|^(write (a |me )?(function|code|test))"
        r"|^(implement|set up|design|build) (a |me |my )?", p)
    if vague_start:
        if len(prompt.split()) <= 6:
            return "vague_ambiguous"
        return "underspecified_tech"

    # Domain knowledge: specific factual questions with domain keywords
    if re.search(r"what (is|are|causes|year|does|'?s)|how (do|does|many|much|to|what|why)"
                 r"|difference between|step.by.step"
                 r"|(sum|area|calculate|percentage|\d+%|\d+\s*\*|\d+\s*\+)"
                 r"|(stack|queue|binary search|data structure|algorithm)"
                 r"|(http|https|sql|bitcoin|docker|ci/cd|machine learning|ai\b|javascript|python)"
                 r"|(iphone|laptop|backup|icloud|authentication|async)"
                 r"|(tides|speed of light|climate|medicine|law|history)", p):
        return "domain_knowledge"

    return "underspecified_tech"


def log_disagreement(result: dict, reason: str = "") -> None:
    """Log a v4-vs-v1 disagreement case for failure-mode analysis.
    Written to logs/disagreement_cases.jsonl (append-only dataset).

    High-impact disagreements are flagged for manual shadow review.
    All disagreements are logged for weekly audit regardless of severity.

    v2.1.1: Added domain_knowledge risk fields (observability only):
      - failure_mode: cluster label
      - confidence_gap: S_v4 - S_v1
      - factuality_risk_flag: HI=1 AND S_v4 > 0.70 AND mode=domain_knowledge
    """
    if not result.get("divergence", False):
        return  # no disagreement, nothing to log

    v4 = result.get("v4", {})
    v1 = result.get("v1", {})
    s_v4 = v4.get("S", 0.0)
    s_v1 = v1.get("S", 0.0)
    dec_v4 = v4.get("decision", "")
    dec_v1 = v1.get("decision", "")

    confidence_gap = round(s_v4 - s_v1, 4)

    # Failure mode classification (observability only — does NOT affect decisions)
    failure_mode = _classify_failure_mode(result.get("prompt", ""))

    # Determine if this is high-impact (needs manual shadow review)
    tier_order = {"accept": 3, "review": 2, "reject": 1}
    tier_diff = abs(tier_order.get(dec_v4, 0) - tier_order.get(dec_v1, 0))

    is_cross_tier = tier_diff > 1  # accept vs reject
    is_accept_near = (dec_v4 == "accept" and s_v4 < 0.80)  # near tau_h=0.70
    is_reject_near = (dec_v4 == "reject" and s_v4 > 0.10)  # near tau_l=0.20

    is_high_impact = (is_cross_tier or is_accept_near or is_reject_near)

    # v2.1.1: domain_knowledge factuality risk flag
    factuality_risk_flag = (
        is_high_impact
        and s_v4 > 0.70
        and failure_mode == "domain_knowledge"
    )

    # Determine the safe action
    if is_high_impact:
        if tier_order.get(dec_v1, 0) < tier_order.get(dec_v4, 0):
            safe_decision = dec_v1
        else:
            safe_decision = "review"
    else:
        safe_decision = dec_v4

    entry = {
        "query_id": result.get("query_id", ""),
        "prompt": result.get("prompt", "")[:500],
        "v4": {"S": s_v4, "kappa": v4.get("kappa", 0), "decision": dec_v4},
        "v1": {"S": s_v1, "kappa": v1.get("kappa", 0), "decision": dec_v1},
        "S_delta": confidence_gap,
        "confidence_gap": confidence_gap,
        "tier_diff": tier_diff,
        "is_cross_tier": is_cross_tier,
        "is_high_impact": is_high_impact,
        "safe_decision": safe_decision,
        "failure_mode": failure_mode,
        "factuality_risk_flag": factuality_risk_flag,
        "reason": reason or ("high_impact_divergence" if is_high_impact else "low_impact_divergence"),
        "monitor_action": _current_monitor_action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        os.makedirs(os.path.dirname(DISAGREEMENT_LOG_PATH), exist_ok=True)
        with open(DISAGREEMENT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass

    return is_high_impact, safe_decision


def capture_prompt(prompt: str, source: str = "api") -> None:
    """Log a real prompt to raw_prompts.jsonl before any model call.
    This is the single data capture point. Only real requests go here."""
    if not prompt or not prompt.strip():
        return
    # Dedup: skip if this exact prompt already exists
    md5 = hashlib.md5(prompt.strip().encode()).hexdigest()
    if os.path.exists(RAW_PROMPTS_PATH):
        try:
            with open(RAW_PROMPTS_PATH, "r") as f:
                for line in f:
                    try:
                        existing = json.loads(line.strip())
                        if hashlib.md5(existing.get("prompt", "").strip().encode()).hexdigest() == md5:
                            return  # already captured
                    except (json.JSONDecodeError, OSError):
                        continue
        except OSError:
            pass
    entry = {
        "prompt": prompt.strip(),
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        os.makedirs(os.path.dirname(RAW_PROMPTS_PATH), exist_ok=True)
        with open(RAW_PROMPTS_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ═══════════════════════════════════════════════════════════════

# Stopwords: high-frequency words that carry no discriminative signal.
# Kept minimal — over-aggressive stopword removal hurts similarity.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "am", "it", "its", "to", "of", "in", "for", "on", "with", "at",
    "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or", "if",
    "while", "that", "this", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "what", "which", "who", "whom", "whose",
})

# Basic synonym mapping: normalizes surface form differences.
# Intentionally small — covers the most common semantic equivalence gaps.
SYNONYMS = {
    "don't": "do not", "doesnt": "does not", "doesn't": "does not",
    "don\u2019t": "do not", "doesn\u2019t": "does not",
    "cant": "can not", "can't": "can not", "cannot": "can not",
    "won't": "will not", "wont": "will not",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "hasn't": "has not", "hasnt": "has not",
    "haven't": "have not", "havent": "have not",
    "didn't": "did not", "didnt": "did not",
    "wouldn't": "would not", "wouldnt": "would not",
    "shouldn't": "should not", "shouldnt": "should not",
    "couldn't": "could not", "couldnt": "could not",
    "it's": "it is", "its": "it is",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "here's": "here is", "what's": "what is",
    "let's": "let us",
    # Semantic normalizations
    "utilize": "use", "utilizes": "uses", "utilizing": "using",
    "approx": "approximately", "approx": "about",
    "info": "information", "tech": "technology",
    "math": "mathematics", "stats": "statistics",
    "diff": "difference", "diffs": "differences",
    "calc": "calculate", "calcs": "calculations",
    "min": "minimum", "max": "maximum",
    "num": "number", "def": "definition", "fn": "function",
    "biz": "business", "org": "organization",
    "rep": "represent", "reps": "representative",
    "gov": "government", "admin": "administration",
    "auth": "authenticate", "req": "request",
    "msg": "message", "err": "error",
    "ok": "okay", "ya": "yes", "yep": "yes", "yeah": "yes",
    "nope": "no", "nah": "no",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of",
    "cause": "because", "cuz": "because",
    # Domain-specific (technical)
    "http": "hypertext transfer protocol",
    "https": "hypertext transfer protocol secure",
    "cpu": "central processing unit",
    "gpu": "graphics processing unit",
    "api": "application programming interface",
    "db": "database", "sql": "structured query language",
    "llm": "large language model",
    "ai": "artificial intelligence", "ml": "machine learning",
}


def _strip_punct(token: str) -> str:
    """Remove all punctuation from a token. Keeps apostrophes for contraction lookup."""
    return "".join(ch for ch in token if ch.isalnum() or ch == "'")


def _depossess(token: str) -> str:
    """Strip possessive 's from token: japan's -> japan."""
    if token.endswith("'s"):
        return token[:-2]
    return token


def normalize_text(text: str) -> str:
    """
    Pre-processing pipeline before TF-IDF:
      1. Lowercase
      2. Strip punctuation (order matters: punct first, then lookup)
      3. Expand contractions via synonym map
      4. Remove stopwords
    """
    text = text.lower().strip()
    tokens = text.split()
    expanded = []
    for tok in tokens:
        # Strip ALL punctuation first (handles commas, periods, etc.)
        clean = _strip_punct(tok)
        if not clean:
            continue
        # Handle possessives: "japan's" -> "japan"
        clean = _depossess(clean)
        # Try contraction/synonym expansion
        replacement = SYNONYMS.get(clean, clean)
        if replacement != clean:
            expanded.extend(replacement.split())
        elif clean not in STOPWORDS:
            expanded.append(clean)
    return " ".join(expanded)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class SurvivalConfig:
    """Configuration for the survival engine. All params tunable."""

    # API
    deepseek_api_key: str = ""
    model: str = "deepseek-chat"
    api_base: str = "https://api.deepseek.com/v1/chat/completions"

    # Alternate provider: Zhipu GLM-4 (set provider="zhipu" to use)
    provider: str = "deepseek"       # "deepseek" | "zhipu"
    zhipu_api_key: str = ""
    zhipu_model: str = "glm-4"
    zhipu_api_base: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    # Perturbation parameters
    n_perturbations: int = 5          # number of perturbed prompts per query
    n_contexts: int = 4               # number of cross-context frames

    # Weights (CALIBRATED — v2.0-survival-stable)
    lambda1: float = 0.5              # weight on local uncertainty
    lambda2: float = 0.5              # weight on global inconsistency
    eps_u: float = 1e-6               # numerical stability

    # Decision thresholds (CALIBRATED — v2.0-survival-stable)
    # τ_h=0.70: bad_accepted=1, good_rejected=0 (strictly dominates v1)
    tau_h: float = 0.70               # accept threshold
    tau_l: float = 0.20               # reject threshold

    # Drift detection
    drift_window: int = 50            # sliding window size
    drift_smooth_window: int = 3      # rolling average for S_dot smoothing
    drift_warn_threshold: float = -0.10  # Ṡ below this = warning

    # ΔG variance weighting (v3 upgrade)
    delta_G_var_weight: float = 1.0  # weight on variance component of ΔG

    # API behavior
    max_tokens: int = 256             # max response tokens per call
    temperature: float = 0.7
    request_delay: float = 0.3        # seconds between API calls (rate limiting)

    # Storage
    survival_log_path: str = os.path.join(DIR, "survival_log.jsonl")
    drift_history_path: str = os.path.join(DIR, "drift_history.jsonl")


# ═══════════════════════════════════════════════════════════════
# PERTURBATION TEMPLATES
# ═══════════════════════════════════════════════════════════════

PERTURBATION_PREFIXES = [
    "",                                     # identity (original)
    "Can you tell me, ",                    # polite wrapper
    "I'm curious: ",                        # conversational
    "Explain simply: ",                     # simplification
    "From a technical perspective, ",        # technical frame
    "Briefly, ",                            # brevity
    "In one sentence, ",                    # extreme brevity
    "Help me understand: ",                 # learner frame
    "What do you think — ",                 # opinion frame
    "Let's say someone asked: ",            # hypothetical
]

# Deterministic context sequence: mixed mild + adversarial.
# Mild contexts: good prompts survive (similar answer, different words).
# Adversarial context: bad prompts diverge (contradictory answers).
# Order matters: first N are used for n_contexts=N evaluation.
CONTEXT_SEQUENCE = [
    ("You are a helpful assistant. Answer normally.", "baseline"),         # control
    ("You are a technical expert. Be precise and detailed.", "expert"),    # mild divergence
    ("Explain this to a complete beginner in simple terms.", "beginner"),  # mild divergence
    ("You must argue the OPPOSITE conclusion. Be convincing.", "adversarial"),  # forced divergence
    ("Summarize your answer in exactly one sentence.", "concise"),         # format stress
    ("Answer using only bullet points. No paragraphs.", "structured"),    # format stress
    ("You are a skeptical reviewer. Challenge everything.", "skeptical"),  # adversarial
    ("You are a data scientist. Focus on quantitative aspects.", "analytical"),  # mild
]


# ═══════════════════════════════════════════════════════════════
# RESULT DATA CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class SurvivalResult:
    """Output of a single survival evaluation."""
    query_id: str = ""
    prompt: str = ""
    timestamp: str = ""

    # Core metrics
    kappa: float = 0.0           # consistency under perturbation
    delta_L: float = 0.0         # local uncertainty
    delta_G: float = 0.0         # global inconsistency
    S: float = 0.0               # survival scalar
    A: float = 0.0               # amplification / brittleness

    # Decision
    decision: str = "reject"     # accept | review | reject

    # Drift
    S_dot: Optional[float] = None       # drift signal
    drift_warning: bool = False

    # Raw data for debugging
    baseline_response: str = ""
    perturbed_responses: list = field(default_factory=list)
    context_responses: list = field(default_factory=list)
    n_api_calls: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Truncate large response fields for JSON logging
        d["baseline_response"] = self.baseline_response[:500]
        d["perturbed_responses"] = [r[:200] for r in self.perturbed_responses[:10]]
        d["context_responses"] = [r[:200] for r in self.context_responses[:10]]
        return d


# ═══════════════════════════════════════════════════════════════
# DEEPSEEK API CLIENT (stdlib only — zero dependencies)
# ═══════════════════════════════════════════════════════════════

class DeepSeekClient:
    """Minimal LLM API client using stdlib urllib only.
    Supports DeepSeek and Zhipu GLM-4 (OpenAI-compatible APIs)."""

    def __init__(self, config: SurvivalConfig):
        self.config = config
        if config.provider == "zhipu":
            self.key = config.zhipu_api_key or config.deepseek_api_key
            self.base = config.zhipu_api_base
            self.model = config.zhipu_model
        else:
            self.key = config.deepseek_api_key
            self.base = config.api_base
            self.model = config.model

    def generate(self, prompt: str, system: str = "") -> str:
        """Send a chat completion request. Returns the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }).encode("utf-8")

        req = Request(
            self.base,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
            method="POST",
        )

        try:
            import socket
            socket.setdefaulttimeout(20)
            with urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(
                f"DeepSeek API error {e.code}: {error_body}"
            ) from e
        except (URLError, socket.timeout, TimeoutError, OSError) as e:
            raise RuntimeError(f"API timeout/error: {type(e).__name__}") from e

    def generate_batch(self, prompts: list[str], system: str = "") -> list[str]:
        """Generate multiple responses with rate limiting."""
        results = []
        for p in prompts:
            results.append(self.generate(p, system))
            time.sleep(self.config.request_delay)
        return results


# ═══════════════════════════════════════════════════════════════
# TEXT SIMILARITY — TF-IDF weighted n-gram cosine
# ═══════════════════════════════════════════════════════════════

def _extract_ngrams(text: str, max_n: int = 2) -> list[str]:
    """Extract 1-grams and 2-grams from PRE-NORMALIZED text.
    Input should already be lowercased, stopword-removed, synonym-expanded."""
    words = text.split()
    ngrams = list(words)  # 1-grams
    for i in range(len(words) - 1):
        ngrams.append(words[i] + " " + words[i + 1])  # 2-grams
    return ngrams


def _compute_tfidf_vectors(texts: list[str]) -> list[dict]:
    """
    Compute TF-IDF weighted vectors for a list of texts.
    Uses 1-grams and 2-grams. IDF is computed across the batch.
    Texts are NORMALIZED before n-gram extraction.
    """
    n = len(texts)
    if n == 0:
        return []

    # Normalize texts then extract n-grams
    normalized = [normalize_text(t) for t in texts]
    all_ngrams = [_extract_ngrams(t) for t in normalized]

    # Compute TF (term frequency) for each text
    tf_lists = []
    for ngrams in all_ngrams:
        tf = {}
        for ng in ngrams:
            tf[ng] = tf.get(ng, 0) + 1
        # Normalize by document length
        length = len(ngrams) if ngrams else 1
        tf = {k: v / length for k, v in tf.items()}
        tf_lists.append(tf)

    # Compute IDF (inverse document frequency)
    doc_freq = {}
    for tf in tf_lists:
        for term in tf:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log((n + 1) / (df + 1)) + 1  # smoothed IDF

    # Compute TF-IDF vectors
    tfidf_vectors = []
    for tf in tf_lists:
        vec = {k: v * idf.get(k, 1.0) for k, v in tf.items()}
        tfidf_vectors.append(vec)

    return tfidf_vectors


def _cosine_sim_tfidf(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    # Dot product (only over shared keys)
    shared = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in shared)
    if dot == 0:
        return 0.0

    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def _pairwise_similarities(texts: list[str]) -> list[float]:
    """Compute all pairwise TF-IDF cosine similarities.
    Texts are normalized (lowercase, stopwords, synonyms) before comparison."""
    if len(texts) < 2:
        return []
    vectors = _compute_tfidf_vectors(texts)
    sims = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim_tfidf(vectors[i], vectors[j]))
    return sims


# ═══════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_kappa(baseline: str, perturbed: list[str]) -> float:
    """
    kappa(x) — consistency under perturbation.

    Average pairwise cosine similarity across all responses
    (baseline + perturbed). Range: [0, 1].

    High kappa = model gives consistent answers regardless of phrasing.
    Low kappa = model is sensitive to input framing.
    """
    all_texts = [baseline] + perturbed
    if len(all_texts) < 2:
        return 1.0

    sims = _pairwise_similarities(all_texts)
    if not sims:
        return 1.0

    return sum(sims) / len(sims)


def compute_delta_L(baseline: str, perturbed: list[str]) -> float:
    """
    delta_L(x) — local uncertainty.

    Variance of pairwise similarities. High variance = model is
    inconsistent in unpredictable ways.

    Proxy for: Var(f(x + delta)) — variance of outputs under perturbation.
    """
    all_texts = [baseline] + perturbed
    if len(all_texts) < 2:
        return 0.0

    sims = _pairwise_similarities(all_texts)
    if len(sims) < 2:
        return 0.0

    mean = sum(sims) / len(sims)
    variance = sum((s - mean) ** 2 for s in sims) / len(sims)

    return variance


# ── v4 METRICS (context-based — zero additional API cost) ──

def compute_kappa_v4(baseline: str, context_responses: list[str]) -> float:
    """
    v4 kappa: mean similarity of baseline vs each context response.
    Uses the already-paid-for context responses instead of perturbation responses.
    """
    if not context_responses:
        return 1.0
    baseline_vec = _compute_tfidf_vectors([baseline])[0]
    sims = []
    for ctx in context_responses:
        ctx_vec = _compute_tfidf_vectors([ctx])[0]
        sims.append(_cosine_sim_tfidf(baseline_vec, ctx_vec))
    return sum(sims) / len(sims) if sims else 1.0


def compute_delta_L_v4(baseline: str, context_responses: list[str]) -> float:
    """
    v4 delta_L: variance of [1.0, sim(baseline, ctx1), ..., sim(baseline, ctxN)].
    Anchored at 1.0 (self-similarity) to ensure non-zero variance even with
    consistent context responses.
    """
    if not context_responses:
        return 0.0
    baseline_vec = _compute_tfidf_vectors([baseline])[0]
    sims = [1.0]  # self-similarity anchor
    for ctx in context_responses:
        ctx_vec = _compute_tfidf_vectors([ctx])[0]
        sims.append(_cosine_sim_tfidf(baseline_vec, ctx_vec))
    mean = sum(sims) / len(sims)
    return sum((s - mean) ** 2 for s in sims) / len(sims)


def compute_delta_G(context_responses: list[str], baseline: str = "",
                       var_weight: float = 1.0) -> float:
    """
    delta_G(x) — global inconsistency (v3: mean + variance).

    Compares each context response against the baseline (normal answer).
    Measures: how much does changing the role/frame alter the answer?

    v3 upgrade: adds variance-of-deviation as second signal.
    Good prompts → low mean deviation + low variance (stable under context shifts).
    Bad prompts  → high mean deviation + high variance (unstable, contradictory).

    Formula:
        deviations = [1 - sim(context_i, baseline)]
        delta_G = mean(deviations) + var_weight * var(deviations)

    If baseline not provided (fallback):
        pairwise among context responses.

    Args:
        context_responses: list of responses under different system prompts
        baseline: the normal (no-context) response
        var_weight: weight on the variance component (default 1.0)
    """
    if not context_responses:
        return 0.0

    # Preferred: compare each context to baseline
    if baseline:
        sims = [_pairwise_similarities([baseline, cr])[0] for cr in context_responses]
        if not sims:
            return 1.0
        # Convert similarities to deviations (1 - sim)
        deviations = [1.0 - s for s in sims]
        mean_dev = sum(deviations) / len(deviations)
        # Variance of deviations: bad prompts have HIGH variance in how contexts distort
        if len(deviations) >= 2:
            var_dev = sum((d - mean_dev) ** 2 for d in deviations) / len(deviations)
        else:
            var_dev = 0.0
        return mean_dev + var_weight * var_dev

    # Fallback: pairwise among contexts
    sims = _pairwise_similarities(context_responses)
    if not sims:
        return 1.0
    avg_sim = sum(sims) / len(sims)
    return 1.0 - avg_sim


def compute_S(kappa: float, delta_L: float, delta_G: float,
              lambda1: float, lambda2: float, eps_u: float = 1e-6) -> float:
    """
    S(x) = kappa / (kappa + lambda1 * delta_L + lambda2 * delta_G + eps_u)

    Range: (0, 1]. High S = robust output. Low S = fragile output.
    """
    denom = kappa + lambda1 * delta_L + lambda2 * delta_G + eps_u
    if denom == 0:
        return 0.0
    return kappa / denom


def compute_A(kappa: float, eps_u: float) -> float:
    """
    A(x) = 1 / (kappa + eps_u)

    Amplification / brittleness signal.
    When kappa -> 0, A spikes -> output is on the edge of failure.
    """
    denom = kappa + eps_u
    if denom == 0:
        return float('inf')
    return 1.0 / denom


def decide_gate(S: float, tau_h: float, tau_l: float) -> str:
    """Three-tier decision gate."""
    if S > tau_h:
        return "accept"
    elif S > tau_l:
        return "review"
    else:
        return "reject"


# ═══════════════════════════════════════════════════════════════
# DRIFT TRACKER
# ═══════════════════════════════════════════════════════════════

class DriftTracker:
    """
    Tracks S(t) over time. Fires warning when Ṡ(t) < threshold.

    Ṡ(t) = S(t) - S(t-1) over sliding window.
    """

    _DRIFT_WARN_THRESHOLD = -0.10

    def __init__(self, config: SurvivalConfig):
        self.config = config
        self.history: list[float] = []
        self._warn_threshold = config.drift_warn_threshold
        self._smooth_window = config.drift_smooth_window

    def update(self, S: float) -> tuple[Optional[float], bool]:
        """
        Add new S value, compute smoothed drift, check warning.

        Uses rolling average of S_dot over drift_smooth_window steps
        to reduce noise. Default window = 3.

        Returns: (S_dot_smooth, warning_fired)
            S_dot_smooth: smoothed drift (None if not enough history)
            warning_fired: True if smoothed S dropped sharply
        """
        self.history.append(S)

        # Trim to window
        if len(self.history) > self.config.drift_window:
            self.history = self.history[-self.config.drift_window:]

        if len(self.history) < 2:
            return None, False

        # Raw S_dot
        raw_dot = self.history[-1] - self.history[-2]

        # Smooth over window: average of last k S_dot values
        k = min(self._smooth_window, len(self.history) - 1)
        dots = [self.history[-i] - self.history[-i - 1] for i in range(1, k + 1)]
        s_dot_smooth = sum(dots) / len(dots)

        warning = s_dot_smooth < self._warn_threshold

        return s_dot_smooth, warning

    def get_recent(self, n: int = 10) -> list[float]:
        """Get the last n S values."""
        return self.history[-n:]

    def get_stats(self) -> dict:
        """Summary statistics of drift history."""
        if not self.history:
            return {"count": 0, "mean": None, "min": None, "max": None,
                    "latest": None, "trend": "none"}

        mean_s = sum(self.history) / len(self.history)
        trend = "stable"
        if len(self.history) >= 3:
            recent = sum(self.history[-3:]) / 3
            older = sum(self.history[:3]) / 3
            if recent - older < -0.05:
                trend = "declining"
            elif recent - older > 0.05:
                trend = "improving"

        return {
            "count": len(self.history),
            "mean": round(mean_s, 4),
            "min": round(min(self.history), 4),
            "max": round(max(self.history), 4),
            "latest": round(self.history[-1], 4),
            "trend": trend,
        }

    def save(self, path: Optional[str] = None):
        """Append latest S to drift history file."""
        if not self.history:
            return
        path = path or self.config.drift_history_path
        if not path:
            return
        entry = {
            "S": self.history[-1],
            "count": len(self.history),
            "mean": sum(self.history) / len(self.history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def load(self, path: Optional[str] = None):
        """Load drift history from file."""
        path = path or self.config.drift_history_path
        if not path or not os.path.exists(path):
            return
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if "S" in entry:
                            self.history.append(entry["S"])
                    except json.JSONDecodeError:
                        pass


# ═══════════════════════════════════════════════════════════════
# PERTURBATION ENGINE
# ═══════════════════════════════════════════════════════════════

def generate_perturbed_prompts(prompt: str, n: int) -> list[str]:
    """Generate n perturbed versions of the prompt. Returns [] if n <= 0."""
    if n <= 0:
        return [prompt]  # still return original for baseline
    # Always include original
    variants = [prompt]

    # Pick n-1 random prefixes
    available = [p for p in PERTURBATION_PREFIXES if p]
    selected = random.sample(available, min(n - 1, len(available)))

    # If we need more variants, reuse with different capitalization
    while len(selected) < n - 1:
        extra = random.choice(available)
        selected.append(extra)

    variants.extend([p + prompt for p in selected[:n - 1]])
    return variants[:n]


def generate_context_prompts(prompt: str, n: int) -> list[tuple[str, str]]:
    """
    Generate n (system_prompt, user_prompt) pairs with different contexts.
    Uses deterministic CONTEXT_SEQUENCE: first N contexts always selected.
    This ensures consistent, reproducible delta_G measurements.
    """
    selected = CONTEXT_SEQUENCE[:n]
    # Pad with adversarial if n exceeds template count
    while len(selected) < n:
        selected.append(CONTEXT_SEQUENCE[3])  # adversarial
    return [(role, prompt) for role, _ in selected[:n]]


# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════

class SurvivalEngine:
    """
    Main survival scalar engine.

    Evaluates a single prompt by:
    1. Generating perturbed variants → measure kappa and delta_L
    2. Running across different contexts → measure delta_G
    3. Computing S(x) and A(x)
    4. Applying decision gate
    5. Tracking drift over time
    """

    def __init__(self, config: SurvivalConfig):
        self.config = config
        self._client = DeepSeekClient(config)
        self._drift = DriftTracker(config)
        self._drift.load()  # restore previous history

    @property
    def client(self) -> DeepSeekClient:
        """Expose client for prompt generation (used by calibrate.py)."""
        return self._client

    @property
    def drift(self) -> DriftTracker:
        """Expose drift tracker."""
        return self._drift

    def evaluate(self, prompt: str, query_id: str = "") -> SurvivalResult:
        """
        v2.1 PRIMARY evaluation — uses v4 (π_E) exclusively.
        No v1 fallback. v1 is frozen and runs only as audit comparator.
        """
        capture_prompt(prompt, source=self.config.provider)
        cfg = self.config
        qid = query_id or hashlib.sha256(prompt.encode()).hexdigest()[:12]
        n_calls = 0

        # ─── Step 1: Baseline ────────────────────────────────
        baseline = self._client.generate(prompt)
        n_calls += 1

        # ─── Step 2: Cross-context → v4 kappa, v4 delta_L, delta_G ─
        n_ctx = max(cfg.n_contexts, 3)
        context_pairs = generate_context_prompts(prompt, n_ctx)
        context_responses = []
        for system, user in context_pairs:
            resp = self._client.generate(user, system=system)
            context_responses.append(resp)
            n_calls += 1

        # v4 metrics (context-based)
        kappa = compute_kappa_v4(baseline, context_responses)
        delta_L = compute_delta_L_v4(baseline, context_responses)
        delta_G = compute_delta_G(context_responses, baseline=baseline,
                                       var_weight=cfg.delta_G_var_weight)

        # ─── Step 4: Compute S, A ──────────────────────────
        S = compute_S(kappa, delta_L, delta_G, cfg.lambda1, cfg.lambda2, cfg.eps_u)
        A = compute_A(kappa, cfg.eps_u)

        # ─── Step 5: Decision gate ──────────────────────────
        decision = decide_gate(S, cfg.tau_h, cfg.tau_l)

        # ─── Step 6: Drift tracking ─────────────────────────
        s_dot, drift_warning = self._drift.update(S)
        self._drift.save()

        # ─── Build result ───────────────────────────────────
        result = SurvivalResult(
            query_id=qid,
            prompt=prompt,
            timestamp=datetime.now(timezone.utc).isoformat(),
            kappa=round(kappa, 4),
            delta_L=round(delta_L, 4),
            delta_G=round(delta_G, 4),
            S=round(S, 4),
            A=round(A, 4),
            decision=decision,
            S_dot=round(s_dot, 4) if s_dot is not None else None,
            drift_warning=drift_warning,
            baseline_response=baseline,
            perturbed_responses=[],  # v2.0: perturbation not used in primary eval
            context_responses=context_responses,
            n_api_calls=n_calls,
        )

        # ─── Log ────────────────────────────────────────────
        self._log(result)

        return result

    def evaluate_shadow(self, prompt: str, query_id: str = "",
                          shadow_log_path: str = "") -> dict:
        """
        Shadow audit mode: v4 decides (π_E), v1 validates (π_S).
        
        v4 is the sole production output policy.
        v1 runs in parallel as frozen comparator for audit purposes.
        
        SAFETY: if v4 and v1 disagree on high-impact outputs:
            → safe_decision overrides v4's decision
            → case logged to logs/disagreement_cases.jsonl
            → flagged as needs_shadow_review=True
        """
        capture_prompt(prompt, source=self.config.provider)
        cfg = self.config
        qid = query_id or hashlib.sha256(prompt.encode()).hexdigest()[:12]
        n_calls = 0

        # Step 1: Baseline
        baseline = self._client.generate(prompt)
        n_calls += 1

        # Step 2: Perturbation responses (for v1)
        n_perturb = max(cfg.n_perturbations, 2)
        perturbed_prompts = generate_perturbed_prompts(prompt, n_perturb)
        perturbed_responses = []
        for pp in perturbed_prompts[1:]:
            resp = self._client.generate(pp)
            perturbed_responses.append(resp)
            n_calls += 1

        # Step 3: Context responses (for v4 + delta_G)
        n_ctx = max(cfg.n_contexts, 3)
        context_pairs = generate_context_prompts(prompt, n_ctx)
        context_responses = []
        for system, user in context_pairs:
            resp = self._client.generate(user, system=system)
            context_responses.append(resp)
            n_calls += 1

        # v1 SCORING (perturbation-based)
        kappa_v1 = compute_kappa(baseline, perturbed_responses)
        delta_L_v1 = compute_delta_L(baseline, perturbed_responses)
        delta_G = compute_delta_G(context_responses, baseline=baseline,
                                      var_weight=cfg.delta_G_var_weight)
        S_v1 = compute_S(kappa_v1, delta_L_v1, delta_G,
                         cfg.lambda1, cfg.lambda2, cfg.eps_u)
        decision_v1 = decide_gate(S_v1, cfg.tau_h, cfg.tau_l)

        # v4 SCORING (context-based)
        kappa_v4 = compute_kappa_v4(baseline, context_responses)
        delta_L_v4 = compute_delta_L_v4(baseline, context_responses)
        S_v4 = compute_S(kappa_v4, delta_L_v4, delta_G,
                         cfg.lambda1, cfg.lambda2, cfg.eps_u)
        decision_v4 = decide_gate(S_v4, cfg.tau_h, cfg.tau_l)

        divergence = decision_v1 != decision_v4
        s_dot, drift_warning = self._drift.update(S_v4)  # v2.1: track v4 drift
        self._drift.save()

        # ─── Disagreement handling (v2.1 safety rule) ─────
        needs_shadow_review = False
        safe_decision = decision_v4  # default: v4 decides
        
        if divergence:
            high_impact, safe_decision = log_disagreement(
                {"query_id": qid, "prompt": prompt, "divergence": True,
                 "v1": {"S": round(S_v1, 4), "kappa": round(kappa_v1, 4), "decision": decision_v1},
                 "v4": {"S": round(S_v4, 4), "kappa": round(kappa_v4, 4), "decision": decision_v4}},
                reason="v4_vs_v1_disagreement"
            ) or (False, decision_v4)
            needs_shadow_review = high_impact
        
        ts = datetime.now(timezone.utc).isoformat()
        result = {
            "query_id": qid, "prompt": prompt, "timestamp": ts,
            "v1": {"kappa": round(kappa_v1, 4), "delta_L": round(delta_L_v1, 4),
                   "delta_G": round(delta_G, 4), "S": round(S_v1, 4),
                   "decision": decision_v1},
            "v4": {"kappa": round(kappa_v4, 4), "delta_L": round(delta_L_v4, 4),
                   "delta_G": round(delta_G, 4), "S": round(S_v4, 4),
                   "decision": decision_v4},
            "divergence": divergence,
            "S_delta": round(S_v4 - S_v1, 4),
            "decision": safe_decision,  # v2.1: safe_decision (may override v4 on high-impact)
            "needs_shadow_review": needs_shadow_review,
            "n_api_calls": n_calls,
        }

        log_path = shadow_log_path or os.path.join(DIR, "shadow_survival_log.jsonl")
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({k: v for k, v in result.items() if k != "decision"}) + "\n")
        except OSError:
            pass

        self._log(SurvivalResult(
            query_id=qid, prompt=prompt, timestamp=ts,
            kappa=round(kappa_v4, 4), delta_L=round(delta_L_v4, 4),
            delta_G=round(delta_G, 4), S=round(S_v4, 4),
            A=round(compute_A(kappa_v4, cfg.eps_u), 4),
            decision=decision_v4, S_dot=round(s_dot, 4) if s_dot is not None else None,
            drift_warning=drift_warning,
            baseline_response=baseline, perturbed_responses=perturbed_responses,
            context_responses=context_responses, n_api_calls=n_calls,
        ))

        return result

    def evaluate_batch(self, prompts: list[str]) -> list[SurvivalResult]:
        """Evaluate multiple prompts sequentially."""
        results = []
        for i, p in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] ", end="", flush=True)
            try:
                r = self.evaluate(p)
                print(f"S={r.S:.3f} {r.decision}")
                results.append(r)
            except Exception as e:
                print(f"FAILED: {e}")
                results.append(SurvivalResult(query_id=f"err_{i}", prompt=p,
                                             decision="error"))
            time.sleep(0.2)
        return results

    def _log(self, result: SurvivalResult):
        """Append result to survival log."""
        path = self.config.survival_log_path
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(result.to_dict(), default=str) + "\n")
        except OSError:
            pass

    def get_drift_stats(self) -> dict:
        """Get drift tracker statistics."""
        return self._drift.get_stats()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Survival Scalar Engine")
    parser.add_argument("prompt", nargs="?", help="Prompt to evaluate")
    parser.add_argument("--n-perturb", type=int, default=5)
    parser.add_argument("--n-contexts", type=int, default=4)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--tau-h", type=float, default=0.70)
    parser.add_argument("--tau-l", type=float, default=0.35)
    args = parser.parse_args()

    cfg = SurvivalConfig(
        deepseek_api_key=key,
        n_perturbations=args.n_perturb,
        n_contexts=args.n_contexts,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        tau_h=args.tau_h,
        tau_l=args.tau_l,
    )
    engine = SurvivalEngine(cfg)

    if args.prompt:
        r = engine.evaluate(args.prompt)
        print(f"\n  S(x) = {r.S:.4f}")
        print(f"  kappa   = {r.kappa:.4f}")
        print(f"  delta_L = {r.delta_L:.4f}")
        print(f"  delta_G = {r.delta_G:.4f}")
        print(f"  A(x)   = {r.A:.4f}")
        print(f"  decision = {r.decision}")
        if r.S_dot is not None:
            print(f"  S_dot    = {r.S_dot:+.4f} {'[DRIFT WARNING]' if r.drift_warning else ''}")
        print(f"  API calls = {r.n_api_calls}")
    else:
        # Quick demo
        print("Survival Engine — no prompt given. Running quick demo...\n")
        demos = [
            "What is the speed of light?",
            "Explain everything about everything comprehensively.",
            "What does she think about this? Is she right?",
        ]
        for p in demos:
            r = engine.evaluate(p, query_id=p[:30])
            print(f"  [{r.decision:>6}] S={r.S:.3f} kappa={r.kappa:.3f} "
                  f"dL={r.delta_L:.3f} dG={r.delta_G:.3f} | {p[:50]}")
