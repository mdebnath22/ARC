"""
semantic_ir_pipeline.py
========================
Executable event-role semantic IR for planning difficulty prediction.

PIPELINE:
  NL description / PDDL → semantic parser (LLM) → event-role IR
  → graph encoder (GNN / transformer) → ARC-style episodic retrieval
  → difficulty prediction

COMPONENTS:
  Step 1: IR schema (Pydantic models)
  Step 2: LLM-based parser with schema validation
  Step 3: PDDL compiler (IR → PDDL actions)
  Step 4: Graph encoder (event-role graph → embedding)
  Step 5: ARC retrieval over semantic graphs

USAGE:
  python semantic_ir_pipeline.py --phase parse --host http://HOST:11434
  python semantic_ir_pipeline.py --phase encode
  python semantic_ir_pipeline.py --phase eval
"""

from __future__ import annotations
import json, re, sys, time, urllib.request, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "planning"
RESULTS = ROOT / "results_planning"; RESULTS.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: IR SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventRole:
    """
    A single event in the planning task.
    Inspired by Paninian kāraka roles: agent, theme, source, goal, instrument.
    Root is the canonical verb (lemmatized, language-agnostic).
    """
    event_id:   str
    root:       str                       # canonical verb lemma e.g. "move", "stack"
    sense:      Optional[str] = None      # domain sense e.g. "transport.package"
    agent:      Optional[str] = None      # who performs the action
    theme:      Optional[str] = None      # what is acted on
    source:     Optional[str] = None      # origin location/state
    goal:       Optional[str] = None      # destination location/state
    instrument: Optional[str] = None     # using what
    condition:  Optional[str] = None      # precondition reference
    purpose:    Optional[str] = None      # why (linked event_id)

    def validate(self):
        """Check for common LLM hallucination patterns."""
        errors = []
        if not self.root:
            errors.append("missing root verb")
        if self.theme is None and self.root in ("move","transport","carry","load","stack","pick-up"):
            errors.append(f"missing theme for action '{self.root}'")
        if self.goal is None and self.root in ("move","transport","fly","drive"):
            errors.append(f"missing goal for action '{self.root}'")
        return errors


@dataclass
class ClauseLink:
    """Temporal/causal link between events."""
    link_type: str   # "sequence" | "purpose" | "cause" | "condition" | "coord"
    from_id:   str
    to_id:     str


@dataclass
class PlanningIR:
    """
    Complete structured IR for a planning instance.
    Replaces FM embeddings with interpretable symbolic structure.
    """
    instance_id:   str
    domain:        str
    events:        List[EventRole]
    links:         List[ClauseLink]
    goal_events:   List[str]              # event_ids that must be achieved
    entities:      Dict[str, str]         # entity_name → entity_type
    n_events:      int = 0
    n_entities:    int = 0
    n_links:       int = 0

    def __post_init__(self):
        self.n_events   = len(self.events)
        self.n_entities = len(self.entities)
        self.n_links    = len(self.links)

    def validate(self):
        """Full structural validation."""
        errors = []
        event_ids = {e.event_id for e in self.events}
        for e in self.events:
            errors.extend([f"event {e.event_id}: {err}" for err in e.validate()])
        for link in self.links:
            if link.from_id not in event_ids:
                errors.append(f"link from unknown event: {link.from_id}")
            if link.to_id not in event_ids:
                errors.append(f"link to unknown event: {link.to_id}")
        for gid in self.goal_events:
            if gid not in event_ids:
                errors.append(f"goal references unknown event: {gid}")
        return errors

    def to_features(self) -> np.ndarray:
        """
        Convert IR to a fixed-dimensional feature vector.
        These replace FM embeddings as ARC's input.
        Interpretable: every dimension has a clear meaning.
        """
        feats = [
            float(self.n_events),
            float(self.n_entities),
            float(self.n_links),
            float(len(self.goal_events)),
            # Role coverage (fraction of events with each role)
            float(sum(1 for e in self.events if e.agent)      / max(self.n_events,1)),
            float(sum(1 for e in self.events if e.theme)      / max(self.n_events,1)),
            float(sum(1 for e in self.events if e.source)     / max(self.n_events,1)),
            float(sum(1 for e in self.events if e.goal)       / max(self.n_events,1)),
            float(sum(1 for e in self.events if e.instrument) / max(self.n_events,1)),
            # Link type counts
            float(sum(1 for l in self.links if l.link_type=="sequence")),
            float(sum(1 for l in self.links if l.link_type=="purpose")),
            float(sum(1 for l in self.links if l.link_type=="cause")),
            float(sum(1 for l in self.links if l.link_type=="condition")),
            # Structural complexity
            float(self.n_links / max(self.n_events, 1)),       # link density
            float(len(set(e.root for e in self.events))),      # action type diversity
            float(len(self.entities) / max(self.n_events, 1)), # entity-to-event ratio
        ]
        return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: LLM-BASED PARSER WITH VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

PARSE_SYSTEM = """You are a semantic parser for planning problems.
Convert the planning problem into a structured event-role representation.

Output ONLY valid JSON matching this exact schema:
{
  "events": [
    {
      "event_id": "e1",
      "root": "move",
      "sense": "transport.package",
      "agent": "truck1",
      "theme": "pkg1",
      "source": "locA",
      "goal": "locB",
      "instrument": null,
      "condition": null,
      "purpose": null
    }
  ],
  "links": [
    {"link_type": "sequence", "from_id": "e1", "to_id": "e2"}
  ],
  "goal_events": ["e3", "e4"],
  "entities": {
    "truck1": "vehicle",
    "pkg1": "package",
    "locA": "location"
  }
}

Rules:
1. Use ONLY these link_type values: sequence, purpose, cause, condition, coord
2. event_id format: e1, e2, e3, ...
3. root must be a canonical verb lemma (move, stack, pick-up, load, fly, drive)
4. Use null for missing roles, never omit keys
5. goal_events lists the event IDs that achieve the goal state
6. Output ONLY the JSON object, no explanation
"""


def call_llm(host, model, prompt, system="", timeout=120):
    payload = json.dumps({
        "model": model,
        "prompt": (f"{system}\n\n{prompt}" if system else prompt),
        "stream": False,
        "options": {"num_predict": 2048, "temperature": 0.0},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "")
    except Exception as e:
        return f"ERROR:{e}"


def parse_instance_to_ir(ep: dict, host: str, model: str) -> Optional[PlanningIR]:
    """
    Parse a PDDL episode into structured IR using constrained LLM generation.
    Validates and repairs the output.
    """
    dom   = ep.get("domain_pddl", "")
    prob  = ep.get("problem_pddl", "")
    desc  = ep.get("description", "")
    dom_name = ep.get("task_type", ep.get("domain", "unknown"))

    prompt = (
        f"Planning domain: {dom_name}\n\n"
        f"Problem description: {desc}\n\n"
        f"PDDL Initial state:\n{ep.get('init_facts', [])}\n\n"
        f"PDDL Goal:\n{ep.get('goal_facts', [])}\n\n"
        f"Available actions (from domain):\n"
        f"{re.findall(r':action\\s+(\\S+)', dom)}\n\n"
        "Parse this into the structured event-role IR format."
    )

    response = call_llm(host, model, prompt, system=PARSE_SYSTEM, timeout=90)

    if response.startswith("ERROR:"): return None

    # Extract JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match: return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        # Attempt repair: extract partial structure
        try:
            # Try to fix common issues
            cleaned = re.sub(r',\s*}', '}', response)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            data = json.loads(re.search(r'\{.*\}', cleaned, re.DOTALL).group())
        except Exception:
            return None

    # Build IR object
    try:
        events = []
        for e_data in data.get("events", []):
            event = EventRole(
                event_id   = str(e_data.get("event_id", f"e{len(events)+1}")),
                root       = str(e_data.get("root", "unknown")).lower(),
                sense      = e_data.get("sense"),
                agent      = e_data.get("agent"),
                theme      = e_data.get("theme"),
                source     = e_data.get("source"),
                goal       = e_data.get("goal"),
                instrument = e_data.get("instrument"),
                condition  = e_data.get("condition"),
                purpose    = e_data.get("purpose"),
            )
            events.append(event)

        links = []
        for l_data in data.get("links", []):
            link_type = l_data.get("link_type", "sequence")
            if link_type not in ("sequence","purpose","cause","condition","coord"):
                link_type = "sequence"
            links.append(ClauseLink(
                link_type = link_type,
                from_id   = str(l_data.get("from_id", "e1")),
                to_id     = str(l_data.get("to_id", "e1")),
            ))

        ir = PlanningIR(
            instance_id = str(ep.get("instance_id", "?")),
            domain      = dom_name,
            events      = events,
            links       = links,
            goal_events = [str(g) for g in data.get("goal_events", [])],
            entities    = {str(k): str(v) for k, v in data.get("entities", {}).items()},
        )

        # Validate
        errors = ir.validate()
        if errors:
            # Non-fatal — log but return the IR anyway
            pass  # errors are acceptable at small scale

        return ir

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: IR → PDDL COMPILER
# ══════════════════════════════════════════════════════════════════════════════

# Canonical root → PDDL action name mapping
ROOT_TO_PDDL = {
    "move":     "drive-truck",
    "drive":    "drive-truck",
    "fly":      "fly-airplane",
    "load":     "load-truck",
    "unload":   "unload-truck",
    "load-air": "load-airplane",
    "unload-air":"unload-airplane",
    "stack":    "stack",
    "unstack":  "unstack",
    "pick":     "pick-up",
    "pick-up":  "pick-up",
    "put":      "put-down",
    "put-down": "put-down",
    "grasp":    "grasp",
    "release":  "release",
    "place":    "place",
    "lift":     "lift",
}


def compile_event_to_pddl(event: EventRole) -> Optional[str]:
    """
    Compile a single EventRole into a PDDL ground action string.
    Returns None if compilation fails.
    """
    action = ROOT_TO_PDDL.get(event.root.lower())
    if action is None: return None

    # Build argument list based on action type
    arg_map = {
        "drive-truck":      [event.agent, event.source, event.goal, None],
        "fly-airplane":     [event.agent, event.source, event.goal],
        "load-truck":       [event.theme, event.agent, event.source],
        "unload-truck":     [event.theme, event.agent, event.goal],
        "load-airplane":    [event.theme, event.agent, event.source],
        "unload-airplane":  [event.theme, event.agent, event.goal],
        "stack":            [event.theme, event.goal],
        "unstack":          [event.theme, event.source],
        "pick-up":          [event.theme],
        "put-down":         [event.theme],
        "grasp":            [event.theme],
        "release":          [event.theme],
        "place":            [event.theme, event.goal],
        "lift":             [event.theme, event.source],
    }

    args = arg_map.get(action, [])
    # Filter None args
    args = [a for a in args if a is not None]
    if not args: return None

    return f"({action} {' '.join(args)})"


def compile_ir_to_pddl_plan(ir: PlanningIR) -> List[str]:
    """Compile full IR into ordered PDDL action sequence."""
    # Order events by links (topological sort via sequence links)
    ordered = _topological_sort(ir)
    plan = []
    for event in ordered:
        pddl_action = compile_event_to_pddl(event)
        if pddl_action: plan.append(pddl_action)
    return plan


def _topological_sort(ir: PlanningIR) -> List[EventRole]:
    """Simple topological sort via sequence links."""
    event_map = {e.event_id: e for e in ir.events}
    seq_links = [(l.from_id, l.to_id) for l in ir.links if l.link_type == "sequence"]

    # Build adjacency
    out_edges = {eid: [] for eid in event_map}
    in_degree  = {eid: 0 for eid in event_map}
    for fr, to in seq_links:
        if fr in out_edges and to in in_degree:
            out_edges[fr].append(to)
            in_degree[to] += 1

    # Kahn's algorithm
    queue = [eid for eid, deg in in_degree.items() if deg == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(event_map[node])
        for succ in out_edges.get(node, []):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # Append any remaining (disconnected events)
    included = {e.event_id for e in result}
    for e in ir.events:
        if e.event_id not in included:
            result.append(e)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GRAPH ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class SemanticGraphEncoder:
    """
    Encodes a PlanningIR into a fixed-dimensional embedding.

    Two modes:
    1. Feature-based: fixed 16-dim interpretable feature vector (fast, no GPU)
    2. GNN-based: message-passing over event-role graph (needs torch-geometric)

    For the ARC paper: use feature-based mode.
    For the follow-up paper: implement full GNN.
    """

    def __init__(self, mode="features", embed_dim=64):
        self.mode      = mode
        self.embed_dim = embed_dim

        if mode == "gnn":
            try:
                import torch
                import torch.nn as nn
                self._init_gnn(embed_dim)
            except ImportError:
                print("torch-geometric not available, falling back to features")
                self.mode = "features"

    def _init_gnn(self, embed_dim):
        """Initialize a simple GNN over the event-role graph."""
        import torch.nn as nn
        # Node feature dim: one-hot role type (8) + entity features (8) = 16
        self.node_dim = 16
        self.gnn_layers = nn.ModuleList([
            nn.Linear(self.node_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
        ])
        self.readout = nn.Linear(embed_dim, embed_dim)

    def encode(self, ir: PlanningIR) -> np.ndarray:
        if self.mode == "features":
            return self._encode_features(ir)
        else:
            return self._encode_gnn(ir)

    def _encode_features(self, ir: PlanningIR) -> np.ndarray:
        """
        Interpretable 16-dim feature vector from IR.
        Each dimension has a clear semantic meaning.
        """
        return ir.to_features()

    def _encode_gnn(self, ir: PlanningIR) -> np.ndarray:
        """
        Graph neural network encoding.
        Nodes = events + entities.
        Edges = role relations + sequence links.
        """
        import torch
        # Build node features
        nodes = []
        for e in ir.events:
            # One-hot encode role presence (8 bits)
            role_bits = [
                float(e.agent is not None),
                float(e.theme is not None),
                float(e.source is not None),
                float(e.goal is not None),
                float(e.instrument is not None),
                float(e.condition is not None),
                float(e.purpose is not None),
                0.0,  # padding
            ]
            # Encode root verb (hash to 8-dim)
            root_hash = [float((hash(e.root) >> i) & 1) for i in range(8)]
            nodes.append(role_bits + root_hash)

        if not nodes:
            return np.zeros(self.embed_dim, dtype=np.float32)

        X = torch.FloatTensor(nodes)  # (n_events, 16)
        # Simple mean-pool over events (placeholder for full GNN)
        embedding = X.mean(0).detach().numpy()
        # Project to embed_dim if needed
        if len(embedding) < self.embed_dim:
            embedding = np.pad(embedding, (0, self.embed_dim - len(embedding)))
        return embedding[:self.embed_dim]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: DIFFICULTY PREDICTION USING IR FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compare_ir_vs_fm_features(ir_list, fm_embeddings, y_ns, domain_name):
    """
    Compare difficulty prediction from IR features vs FM embeddings.
    Uses XGBoost as a controlled classifier (no architecture differences).
    """
    import xgboost as xgb
    from scipy import stats
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    encoder = SemanticGraphEncoder(mode="features")
    X_ir = np.array([encoder.encode(ir) for ir in ir_list])
    X_fm = np.array(fm_embeddings)

    # Normalise
    X_ir = StandardScaler().fit_transform(X_ir)
    X_fm = StandardScaler().fit_transform(X_fm)

    results = {}
    for name, X in [("IR features", X_ir), ("FM embeddings", X_fm),
                    ("IR + FM", np.hstack([X_ir, X_fm]))]:
        clf = xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0,
                                random_state=42)
        preds = cross_val_predict(clf, X, y_ns, cv=5)
        rho = abs(float(stats.spearmanr(preds, y_ns)[0]))
        results[name] = rho
        print(f"  {name:<20}: |ρ| = {rho:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_parse_phase(host, model="qwen2.5:72b", n_per_domain=50):
    """Parse planning instances into IR using LLM."""
    print(f"\n{'='*60}")
    print("PHASE 1: Parsing instances to semantic IR")
    print(f"  Model: {model}  N: {n_per_domain} per domain")
    print('='*60)

    eps_by_dom = {}
    for e in json.loads((DATA/"episodes.json").read_text()):
        eps_by_dom.setdefault(e.get("task_type", e.get("domain","")), []).append(e)

    all_irs = {}
    test_domains = ["blocksworld", "logistics", "mystery_blocksworld"]

    for dom in test_domains:
        eps  = eps_by_dom.get(dom, [])[:n_per_domain]
        irs  = []
        encoder = SemanticGraphEncoder(mode="features")

        print(f"\n  {dom} ({len(eps)} instances):")
        for i, ep in enumerate(eps):
            ir = parse_instance_to_ir(ep, host, model)
            if ir:
                errors = ir.validate()
                irs.append({
                    "instance_id": ep.get("instance_id"),
                    "ir": asdict(ir),
                    "features": encoder.encode(ir).tolist(),
                    "validation_errors": errors,
                    "n_events": ir.n_events,
                    "n_entities": ir.n_entities,
                })
                print(f"    [{i+1}/{len(eps)}] events={ir.n_events} "
                      f"entities={ir.n_entities} "
                      f"errors={len(errors)}", end="\r")
            else:
                print(f"    [{i+1}/{len(eps)}] PARSE FAILED", end="\r")

        print(f"\n  Parsed: {len(irs)}/{len(eps)}")
        all_irs[dom] = irs

    out = RESULTS / "semantic_ir_parsed.json"
    out.write_text(json.dumps(all_irs, indent=2))
    print(f"\n  Saved → {out}")
    return all_irs


def run_encode_phase():
    """Encode IR features and compare vs FM embeddings for difficulty prediction."""
    print(f"\n{'='*60}")
    print("PHASE 2: Feature comparison (IR vs FM vs IR+FM)")
    print('='*60)

    ir_path = RESULTS/"semantic_ir_parsed.json"
    if not ir_path.exists():
        print("  Run --phase parse first"); return

    ir_data = json.loads(ir_path.read_text())

    # Load ground truth labels
    import importlib.util
    spec6 = importlib.util.spec_from_file_location("s6", ROOT/"plan_step6_pddlinst_gate.py")
    s6    = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    X_surf, X_fm, tt, y_s, y_ns, _ = s6.load_data(data_dir=DATA)

    encoder = SemanticGraphEncoder(mode="features")
    print(f"\n  {'Domain':<18}  {'IR feats':>10}  {'FM embed':>10}  {'IR+FM':>10}")
    print("  "+"-"*50)

    for dom in ["blocksworld","logistics","mystery_blocksworld"]:
        entries = ir_data.get(dom, [])
        if not entries: continue

        iids  = [int(e["instance_id"]) for e in entries]
        mask  = tt == dom
        ns_q  = y_ns[mask].astype(float)
        fm_q  = X_fm[mask]

        # Align by instance_id
        from dataclasses import fields as dc_fields
        ir_feats = np.array([e["features"] for e in entries])
        fm_feats = np.array([fm_q[iid] for iid in iids if iid < len(fm_q)])
        y_sub    = np.array([ns_q[iid] for iid in iids if iid < len(ns_q)])
        n        = min(len(ir_feats), len(fm_feats), len(y_sub))

        results = compare_ir_vs_fm_features(
            [None]*n,  # placeholder (features already computed)
            fm_feats[:n],
            y_sub[:n],
            dom
        )
        dl=dom[:3].upper()
        # Override with pre-computed IR features
        from scipy import stats
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        X_ir_s = StandardScaler().fit_transform(ir_feats[:n])
        clf = xgb.XGBRegressor(n_estimators=100,max_depth=4,verbosity=0,random_state=42)
        from sklearn.model_selection import cross_val_predict
        p_ir = cross_val_predict(clf, X_ir_s, y_sub[:n], cv=5)
        rho_ir = abs(float(stats.spearmanr(p_ir, y_sub[:n])[0]))
        print(f"  {dom:<18}  IR={rho_ir:.3f}  FM={results.get('FM embeddings',0):.3f}  "
              f"IR+FM={results.get('IR + FM',0):.3f}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["parse","encode","all"], default="parse")
    p.add_argument("--host",  default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:72b")
    p.add_argument("--n",     type=int, default=50,
                   help="Instances per domain to parse")
    args = p.parse_args()

    if args.phase in ("parse","all"):
        run_parse_phase(args.host, args.model, args.n)

    if args.phase in ("encode","all"):
        run_encode_phase()


if __name__ == "__main__":
    main()
