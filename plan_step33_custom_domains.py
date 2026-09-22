"""
plan_step33_custom_domains.py
==============================
Creates 5 completely custom non-IPC PDDL domains from scratch,
generates 200 instances each, runs BFS, evaluates ARC vs |O|.

Domains:
  1. Robot Arm Stacking    — multi-table manipulation
  2. Coffee Shop           — order preparation and serving
  3. Package Sorting       — warehouse bin sorting
  4. Rescue Mission        — agent rescues victims across rooms
  5. Library Books         — books returned to correct shelves

All pure STRIPS, typed, pyperplan-compatible.

USAGE:
  python plan_step33_custom_domains.py --generate   # write domains + problems
  python plan_step33_custom_domains.py --evaluate   # run BFS + ARC
  python plan_step33_custom_domains.py --all        # both
"""

from __future__ import annotations
import argparse, importlib.util, json, pickle, random, re, signal, sys
import tempfile, warnings
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent
OOD    = ROOT / "data" / "ood_eval2"
RES    = ROOT / "results_planning"; RES.mkdir(exist_ok=True)
CKPT   = ROOT / "checkpoints_planning"
TRAIN  = ["depot","rovers","satellite"]

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

DOMAINS = {}

# ─────────────────────────────────────────────────────────────────────────
# 1. Robot Arm Stacking
# ─────────────────────────────────────────────────────────────────────────
DOMAINS["robot_arm"] = {
"domain": """(define (domain robot-arm-stacking)
  (:requirements :strips :typing)
  (:types block table - object)
  (:predicates
    (on ?b1 - block ?b2 - block)
    (on-table ?b - block ?t - table)
    (clear ?b - block)
    (table-free ?t - table)
    (holding ?b - block)
    (arm-empty)
  )
  (:action pick-from-table
    :parameters (?b - block ?t - table)
    :precondition (and (on-table ?b ?t) (clear ?b) (arm-empty))
    :effect (and (holding ?b) (table-free ?t)
                 (not (on-table ?b ?t)) (not (clear ?b)) (not (arm-empty))))
  (:action pick-from-block
    :parameters (?b1 - block ?b2 - block ?t - table)
    :precondition (and (on ?b1 ?b2) (clear ?b1) (arm-empty) (on-table ?b2 ?t))
    :effect (and (holding ?b1) (clear ?b2)
                 (not (on ?b1 ?b2)) (not (clear ?b1)) (not (arm-empty))))
  (:action place-on-table
    :parameters (?b - block ?t - table)
    :precondition (and (holding ?b) (table-free ?t))
    :effect (and (on-table ?b ?t) (clear ?b) (arm-empty) (not (table-free ?t))
                 (not (holding ?b))))
  (:action stack
    :parameters (?b1 - block ?b2 - block ?t - table)
    :precondition (and (holding ?b1) (clear ?b2) (on-table ?b2 ?t))
    :effect (and (on ?b1 ?b2) (clear ?b1) (arm-empty)
                 (not (holding ?b1)) (not (clear ?b2))))
)""",

"generate": lambda n, seed: _gen_robot_arm(n, seed),
"desc": "Custom robotic arm manipulation: pick/place blocks across tables"
}

def _gen_robot_arm(n_blocks, seed):
    rng = random.Random(seed)
    n_tables = max(2, n_blocks // 3)
    blocks = [f"b{i}" for i in range(1, n_blocks+1)]
    tables = [f"t{i}" for i in range(1, n_tables+1)]

    # Random initial state
    block_table = {b: rng.choice(tables) for b in blocks}
    # Some blocks stacked
    stacked_on = {}  # b1 -> b2 (b1 is on b2)
    tops = list(blocks); rng.shuffle(tops)
    n_stacks = rng.randint(0, n_blocks // 3)
    used_base = set(); used_top = set()
    for i in range(n_stacks):
        if i+1 >= len(tops): break
        top_b, base_b = tops[i], tops[i+1]
        if top_b not in used_top and base_b not in used_base:
            stacked_on[top_b] = base_b
            block_table[top_b] = block_table[base_b]
            used_top.add(top_b); used_base.add(base_b)

    init = ["(arm-empty)"]
    directly_on_table = {b for b in blocks if b not in stacked_on}
    tables_with_blocks = {block_table[b] for b in directly_on_table}
    for b in directly_on_table:
        init.append(f"(on-table {b} {block_table[b]})")
    for b, base in stacked_on.items():
        init.append(f"(on {b} {base})")
    under = set(stacked_on.values())
    for b in blocks:
        if b not in under:
            init.append(f"(clear {b})")
    for t in tables:
        if t not in tables_with_blocks:
            init.append(f"(table-free {t})")

    # Goal: rearrange some blocks
    goal_t = tables[0]
    goal_blocks = rng.sample(blocks, min(rng.randint(2, n_blocks), n_blocks))
    goal = []
    for i in range(0, len(goal_blocks)-1, 2):
        goal.append(f"(on {goal_blocks[i]} {goal_blocks[i+1]})")
        goal.append(f"(on-table {goal_blocks[i+1]} {goal_t})")
    if not goal:
        goal.append(f"(on-table {goal_blocks[0]} {goal_t})")

    return _pddl_problem(f"robot-arm-n{n_blocks}-s{seed}", "robot-arm-stacking",
                         blocks, "block", tables, "table", init, goal)


# ─────────────────────────────────────────────────────────────────────────
# 2. Coffee Shop
# ─────────────────────────────────────────────────────────────────────────
DOMAINS["coffee_shop"] = {
"domain": """(define (domain coffee-shop)
  (:requirements :strips :typing)
  (:types order ingredient station - object)
  (:predicates
    (order-at ?o - order ?s - station)
    (ingredient-at ?i - ingredient ?s - station)
    (order-needs ?o - order ?i - ingredient)
    (order-ready ?o - order)
    (ingredient-used ?i - ingredient)
    (station-busy ?s - station)
    (order-served ?o - order)
  )
  (:action move-ingredient
    :parameters (?i - ingredient ?from - station ?to - station)
    :precondition (and (ingredient-at ?i ?from) (not (station-busy ?to)))
    :effect (and (ingredient-at ?i ?to)
                 (not (ingredient-at ?i ?from))))
  (:action prepare-order
    :parameters (?o - order ?i - ingredient ?s - station)
    :precondition (and (order-at ?o ?s) (order-needs ?o ?i)
                       (ingredient-at ?i ?s) (not (ingredient-used ?i)))
    :effect (and (ingredient-used ?i) (order-ready ?o)))
  (:action serve-order
    :parameters (?o - order ?s - station ?dest - station)
    :precondition (and (order-at ?o ?s) (order-ready ?o)
                       (not (station-busy ?dest)))
    :effect (and (order-served ?o) (order-at ?o ?dest) (station-busy ?dest)
                 (not (order-at ?o ?s))))
)""",

"generate": lambda n, seed: _gen_coffee_shop(n, seed),
"desc": "Custom coffee shop: prepare and serve orders using ingredients at stations"
}

def _gen_coffee_shop(n_orders, seed):
    rng = random.Random(seed)
    n_ingredients = max(2, n_orders)
    n_stations = max(2, n_orders // 2 + 1)
    orders = [f"ord{i}" for i in range(1, n_orders+1)]
    ingredients = [f"ing{i}" for i in range(1, n_ingredients+1)]
    stations = [f"sta{i}" for i in range(1, n_stations+1)]

    order_station = {o: rng.choice(stations) for o in orders}
    ing_station = {i: rng.choice(stations) for i in ingredients}
    order_needs = {o: rng.choice(ingredients) for o in orders}

    init = []
    for o, s in order_station.items():
        init.append(f"(order-at {o} {s})")
    for i, s in ing_station.items():
        init.append(f"(ingredient-at {i} {s})")
    for o, i in order_needs.items():
        init.append(f"(order-needs {o} {i})")

    goal_orders = rng.sample(orders, min(rng.randint(1, n_orders), n_orders))
    goal = [f"(order-served {o})" for o in goal_orders]

    all_objs = [(orders,"order"),(ingredients,"ingredient"),(stations,"station")]
    return _pddl_problem_multi(f"coffee-n{n_orders}-s{seed}", "coffee-shop",
                               all_objs, init, goal)


# ─────────────────────────────────────────────────────────────────────────
# 3. Package Sorting
# ─────────────────────────────────────────────────────────────────────────
DOMAINS["package_sort"] = {
"domain": """(define (domain package-sorting)
  (:requirements :strips :typing)
  (:types package bin conveyor - object)
  (:predicates
    (package-in-bin ?p - package ?b - bin)
    (package-on-conveyor ?p - package ?c - conveyor)
    (bin-has-space ?b - bin)
    (conveyor-active ?c - conveyor)
    (package-sorted ?p - package)
    (correct-bin ?p - package ?b - bin)
  )
  (:action load-to-conveyor
    :parameters (?p - package ?b - bin ?c - conveyor)
    :precondition (and (package-in-bin ?p ?b) (conveyor-active ?c))
    :effect (and (package-on-conveyor ?p ?c)
                 (bin-has-space ?b)
                 (not (package-in-bin ?p ?b))))
  (:action sort-package
    :parameters (?p - package ?c - conveyor ?b - bin)
    :precondition (and (package-on-conveyor ?p ?c) (bin-has-space ?b)
                       (correct-bin ?p ?b))
    :effect (and (package-in-bin ?p ?b) (package-sorted ?p)
                 (not (package-on-conveyor ?p ?c))
                 (not (bin-has-space ?b))))
  (:action activate-conveyor
    :parameters (?c - conveyor)
    :precondition (not (conveyor-active ?c))
    :effect (conveyor-active ?c))
)""",

"generate": lambda n, seed: _gen_package_sort(n, seed),
"desc": "Custom warehouse package sorting: load packages and route to correct bins"
}

def _gen_package_sort(n_packages, seed):
    rng = random.Random(seed)
    n_bins = max(2, n_packages // 2)
    n_conveyors = max(1, n_packages // 3)
    packages = [f"pkg{i}" for i in range(1, n_packages+1)]
    bins = [f"bin{i}" for i in range(1, n_bins+1)]
    conveyors = [f"conv{i}" for i in range(1, n_conveyors+1)]

    pkg_bin = {p: rng.choice(bins) for p in packages}  # current location
    correct = {p: rng.choice(bins) for p in packages}   # target bin
    active_convs = rng.sample(conveyors, max(1, len(conveyors)//2))

    init = []
    bins_with_pkg = {}
    for p, b in pkg_bin.items():
        init.append(f"(package-in-bin {p} {b})")
        bins_with_pkg[b] = bins_with_pkg.get(b, 0) + 1
    for p, b in correct.items():
        init.append(f"(correct-bin {p} {b})")
    for b in bins:
        if bins_with_pkg.get(b, 0) < 2:
            init.append(f"(bin-has-space {b})")
    for c in active_convs:
        init.append(f"(conveyor-active {c})")

    n_goals = rng.randint(1, n_packages)
    goal_pkgs = rng.sample(packages, n_goals)
    goal = [f"(package-sorted {p})" for p in goal_pkgs]

    all_objs = [(packages,"package"),(bins,"bin"),(conveyors,"conveyor")]
    return _pddl_problem_multi(f"sort-n{n_packages}-s{seed}", "package-sorting",
                               all_objs, init, goal)


# ─────────────────────────────────────────────────────────────────────────
# 4. Rescue Mission
# ─────────────────────────────────────────────────────────────────────────
DOMAINS["rescue"] = {
"domain": """(define (domain rescue-mission)
  (:requirements :strips :typing)
  (:types agent victim room supply - object)
  (:predicates
    (agent-at ?a - agent ?r - room)
    (victim-at ?v - victim ?r - room)
    (supply-at ?s - supply ?r - room)
    (connected ?r1 - room ?r2 - room)
    (victim-rescued ?v - victim)
    (agent-has-supply ?a - agent ?s - supply)
    (victim-needs-supply ?v - victim ?s - supply)
    (room-cleared ?r - room)
  )
  (:action move
    :parameters (?a - agent ?from - room ?to - room)
    :precondition (and (agent-at ?a ?from) (connected ?from ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from))))
  (:action pick-supply
    :parameters (?a - agent ?s - supply ?r - room)
    :precondition (and (agent-at ?a ?r) (supply-at ?s ?r))
    :effect (and (agent-has-supply ?a ?s) (not (supply-at ?s ?r))))
  (:action rescue-victim
    :parameters (?a - agent ?v - victim ?s - supply ?r - room)
    :precondition (and (agent-at ?a ?r) (victim-at ?v ?r)
                       (agent-has-supply ?a ?s) (victim-needs-supply ?v ?s))
    :effect (and (victim-rescued ?v) (not (victim-at ?v ?r))))
)""",

"generate": lambda n, seed: _gen_rescue(n, seed),
"desc": "Custom rescue mission: navigate rooms, collect supplies, rescue victims"
}

def _gen_rescue(n_victims, seed):
    rng = random.Random(seed)
    n_rooms = max(3, n_victims + 1)
    n_agents = max(1, n_victims // 3)
    n_supplies = max(n_victims, 2)
    agents = [f"ag{i}" for i in range(1, n_agents+1)]
    victims = [f"vic{i}" for i in range(1, n_victims+1)]
    rooms = [f"room{i}" for i in range(1, n_rooms+1)]
    supplies = [f"sup{i}" for i in range(1, n_supplies+1)]

    # Random room connectivity (chain + some extras)
    connections = [(rooms[i], rooms[i+1]) for i in range(len(rooms)-1)]
    for _ in range(n_rooms // 2):
        r1, r2 = rng.sample(rooms, 2)
        if (r1,r2) not in connections and (r2,r1) not in connections:
            connections.append((r1, r2))

    agent_room = {a: rng.choice(rooms) for a in agents}
    victim_room = {v: rng.choice(rooms) for v in victims}
    supply_room = {s: rng.choice(rooms) for s in supplies}
    victim_needs = {v: rng.choice(supplies) for v in victims}

    init = []
    for a, r in agent_room.items(): init.append(f"(agent-at {a} {r})")
    for v, r in victim_room.items(): init.append(f"(victim-at {v} {r})")
    for s, r in supply_room.items(): init.append(f"(supply-at {s} {r})")
    for r1, r2 in connections:
        init.append(f"(connected {r1} {r2})")
        init.append(f"(connected {r2} {r1})")
    for v, s in victim_needs.items(): init.append(f"(victim-needs-supply {v} {s})")

    goal_v = rng.sample(victims, rng.randint(1, n_victims))
    goal = [f"(victim-rescued {v})" for v in goal_v]

    all_objs = [(agents,"agent"),(victims,"victim"),(rooms,"room"),(supplies,"supply")]
    return _pddl_problem_multi(f"rescue-n{n_victims}-s{seed}", "rescue-mission",
                               all_objs, init, goal)


# ─────────────────────────────────────────────────────────────────────────
# 5. Library Books
# ─────────────────────────────────────────────────────────────────────────
DOMAINS["library"] = {
"domain": """(define (domain library-books)
  (:requirements :strips :typing)
  (:types book shelf cart patron - object)
  (:predicates
    (book-on-shelf ?b - book ?s - shelf)
    (book-on-cart ?b - book ?c - cart)
    (book-with-patron ?b - book ?p - patron)
    (shelf-has-space ?s - shelf)
    (cart-at-shelf ?c - cart ?s - shelf)
    (correct-shelf ?b - book ?s - shelf)
    (book-returned ?b - book)
    (patron-at-shelf ?p - patron ?s - shelf)
  )
  (:action checkout-book
    :parameters (?b - book ?s - shelf ?p - patron)
    :precondition (and (book-on-shelf ?b ?s) (patron-at-shelf ?p ?s))
    :effect (and (book-with-patron ?b ?p) (shelf-has-space ?s)
                 (not (book-on-shelf ?b ?s))))
  (:action return-to-cart
    :parameters (?b - book ?p - patron ?c - cart ?s - shelf)
    :precondition (and (book-with-patron ?b ?p) (cart-at-shelf ?c ?s))
    :effect (and (book-on-cart ?b ?c) (not (book-with-patron ?b ?p))))
  (:action shelve-book
    :parameters (?b - book ?c - cart ?s - shelf)
    :precondition (and (book-on-cart ?b ?c) (shelf-has-space ?s)
                       (correct-shelf ?b ?s) (cart-at-shelf ?c ?s))
    :effect (and (book-on-shelf ?b ?s) (book-returned ?b)
                 (not (book-on-cart ?b ?c)) (not (shelf-has-space ?s))))
  (:action move-cart
    :parameters (?c - cart ?from - shelf ?to - shelf)
    :precondition (and (cart-at-shelf ?c ?from) (shelf-has-space ?to))
    :effect (and (cart-at-shelf ?c ?to) (not (cart-at-shelf ?c ?from))))
)""",

"generate": lambda n, seed: _gen_library(n, seed),
"desc": "Custom library: check out books, return to carts, shelve at correct location"
}

def _gen_library(n_books, seed):
    rng = random.Random(seed)
    n_shelves = max(2, n_books // 2)
    n_carts = max(1, n_books // 4)
    n_patrons = max(1, n_books // 3)
    books = [f"bk{i}" for i in range(1, n_books+1)]
    shelves = [f"sh{i}" for i in range(1, n_shelves+1)]
    carts = [f"cart{i}" for i in range(1, n_carts+1)]
    patrons = [f"pat{i}" for i in range(1, n_patrons+1)]

    book_shelf = {b: rng.choice(shelves) for b in books}
    correct_shelf = {b: rng.choice(shelves) for b in books}
    cart_shelf = {c: rng.choice(shelves) for c in carts}
    patron_shelf = {p: rng.choice(shelves) for p in patrons}

    init = []
    shelves_full = {}
    for b, s in book_shelf.items():
        init.append(f"(book-on-shelf {b} {s})")
        shelves_full[s] = shelves_full.get(s, 0) + 1
    for s in shelves:
        if shelves_full.get(s, 0) < 2:
            init.append(f"(shelf-has-space {s})")
    for c, s in cart_shelf.items():
        init.append(f"(cart-at-shelf {c} {s})")
    for p, s in patron_shelf.items():
        init.append(f"(patron-at-shelf {p} {s})")
    for b, s in correct_shelf.items():
        init.append(f"(correct-shelf {b} {s})")

    n_goals = rng.randint(1, n_books)
    goal_books = rng.sample(books, n_goals)
    goal = [f"(book-returned {b})" for b in goal_books]

    all_objs = [(books,"book"),(shelves,"shelf"),(carts,"cart"),(patrons,"patron")]
    return _pddl_problem_multi(f"library-n{n_books}-s{seed}", "library-books",
                               all_objs, init, goal)


# ═══════════════════════════════════════════════════════════════════════════
# PDDL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _pddl_problem(name, domain, objs1, type1, objs2, type2, init, goal):
    obj_str = " ".join(objs1) + f" - {type1}\n    " + " ".join(objs2) + f" - {type2}"
    return f"""(define (problem {name})
  (:domain {domain})
  (:objects
    {obj_str}
  )
  (:init
    {chr(10).join("    " + f for f in init)}
  )
  (:goal (and
    {chr(10).join("    " + f for f in goal)}
  ))
)
"""

def _pddl_problem_multi(name, domain, type_groups, init, goal):
    obj_lines = "\n    ".join(
        " ".join(objs) + f" - {t}" for objs, t in type_groups if objs
    )
    return f"""(define (problem {name})
  (:domain {domain})
  (:objects
    {obj_lines}
  )
  (:init
    {chr(10).join("    " + f for f in init)}
  )
  (:goal (and
    {chr(10).join("    " + f for f in goal)}
  ))
)
"""


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE
# ═══════════════════════════════════════════════════════════════════════════

def generate_all():
    print("Generating custom non-IPC domains...")
    sizes = list(range(3, 13))  # n_objects 3..12, 200 instances each
    for dom_name, dom_data in DOMAINS.items():
        dom_dir = OOD / dom_name
        dom_dir.mkdir(parents=True, exist_ok=True)
        (dom_dir / "domain.pddl").write_text(dom_data["domain"])
        idx = 0
        for n in sizes:
            for seed in range(20):  # 10 sizes × 20 seeds = 200
                idx += 1
                try:
                    prob = dom_data["generate"](n, seed)
                    (dom_dir / f"p{idx}.pddl").write_text(prob)
                except Exception as e:
                    print(f"  {dom_name} n={n} seed={seed}: {e}")
        print(f"  {dom_name}: {idx} instances → {dom_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════════════

def run_bfs(dom_txt, prob_txt, timeout=10):
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding
    from pyperplan.search.breadth_first_search import breadth_first_search
    from pyperplan.search.enforced_hillclimbing_search import enforced_hillclimbing_search
    from pyperplan.heuristics.relaxation import hFFHeuristic

    def _to(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _to); signal.alarm(timeout)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dp = Path(tmp)/"domain.pddl"; pp = Path(tmp)/"problem.pddl"
            dp.write_text(dom_txt); pp.write_text(prob_txt)
            from pyperplan.pddl.parser import Parser
            from pyperplan import grounding
            parser = Parser(str(dp), str(pp))
            task = grounding.ground(parser.parse_problem(parser.parse_domain()))
            sol = breadth_first_search(task)
            if sol is None:
                try: sol = enforced_hillclimbing_search(task, hFFHeuristic(task))
                except: pass
            signal.alarm(0)
            return len(sol) if sol else -1
    except (TimeoutError, Exception):
        signal.alarm(0); return -1


def extract_xs(dom_txt, prob_txt):
    obj_sec = re.search(r':objects([^)]*)\)', prob_txt, re.DOTALL|re.I)
    n_obj = len(re.findall(r'\b\w[\w-]*\b', obj_sec.group(1))) if obj_sec else 1
    n_obj = max(n_obj, 1)
    init_txt = prob_txt.split(':goal')[0].split(':init')[-1] if ':init' in prob_txt else ''
    goal_txt = prob_txt.split(':goal')[-1][:500] if ':goal' in prob_txt else ''
    init_f = re.findall(r'\(\w[\w-]*[^)]*\)', init_txt)
    goal_f = re.findall(r'\(\w[\w-]*[^)]*\)', goal_txt)
    ops = re.findall(r':action\s+\S+', dom_txt)
    n_init = len(init_f); n_goal = len(goal_f); n_ops = len(ops)
    pred_c = {}
    for p in init_f:
        name = p.strip('()').split()[0] if p.strip('()').split() else 'x'
        pred_c[name] = pred_c.get(name, 0) + 1
    pv = list(pred_c.values()) or [0]
    feats = np.array([
        float(n_obj), float(n_goal), float(n_init), float(n_ops), float(n_goal),
        float(np.mean(pv)), float(np.std(pv) if len(pv)>1 else 0), float(np.max(pv)),
        float(len(pred_c)), float(n_ops%2), float(n_goal/n_obj), float(n_init/max(n_goal,1)),
        float(n_obj/max(len(pred_c),1)), float(n_goal*np.mean(pv)),
        float(len(pred_c)/max(n_ops,1)), float(len(pred_c)/n_obj),
        float(n_init/n_obj**2), float(n_goal/n_obj**2),
        float(len(pred_c)/max(n_ops,1)), float(np.mean(pv)*n_goal),
        float(n_obj*np.mean(pv)), float(n_goal/max(n_init,1)),
        float(n_init/max(n_ops,1)), float(n_goal/max(n_init,1)),
        float(n_obj/max(n_goal,1)), float(len(pred_c)/n_obj),
        float(n_init/n_obj), float(n_ops/n_obj),
        float(n_obj*n_goal/max(n_init,1)), float(0.0),
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0, posinf=0, neginf=0), n_obj


def evaluate_all():
    print("\nEvaluating ARC on custom non-IPC domains...")

    # Load model
    spec6 = importlib.util.spec_from_file_location("s6", ROOT/"plan_step6_pddlinst_gate.py")
    s6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(s6)
    spec3 = importlib.util.spec_from_file_location("s3", ROOT/"plan_step3_guru.py")
    s3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(s3)
    X_surf, X_fm, tt, _, y_ns, _ = s6.load_data(data_dir=ROOT/"data"/"planning")
    tr = np.isin(tt, TRAIN)
    sc_s = StandardScaler().fit(X_surf[tr]); sc_f = StandardScaler().fit(X_fm[tr])
    pp = Pipeline([("pca", PCA(20)), ("r", Ridge(1.0))])
    pp.fit(sc_s.transform(X_surf[tr]), sc_f.transform(X_fm[tr]))

    def tfm(Xs, Xe):
        Xs_n = sc_s.transform(Xs); Xe_n = sc_f.transform(Xe)
        return Xs_n, Xe_n, Xe_n - pp.predict(Xs_n)

    rng = np.random.default_rng(42)
    sidx = rng.choice(tr.sum(), min(60, tr.sum()), replace=False)
    Xs_tr, Xe_tr, _ = tfm(X_surf[tr], X_fm[tr])
    S_s = torch.FloatTensor(Xs_tr[sidx]); S_f = torch.FloatTensor(Xe_tr[sidx])
    S_V = torch.FloatTensor(np.hstack([Xs_tr[sidx], Xe_tr[sidx]]))
    ck = torch.load(CKPT/"guru_baseline_5000ep.pt", map_location="cpu")
    model = s3.PlanningGURU(30, X_fm.shape[1])
    model.load_state_dict(ck["model"]); model.eval()

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-mpnet-base-v2")

    results = {}
    print(f"\n{'Domain':<16}  {'N':>4}  {'BFS%':>6}  {'ARC|ρ|':>8}  {'|O||ρ|':>8}  {'Δ':>6}")
    print("-"*58)

    for dom_name, dom_data in DOMAINS.items():
        dom_dir = OOD / dom_name
        dom_path = dom_dir / "domain.pddl"
        if not dom_path.exists():
            print(f"  {dom_name}: not generated yet"); continue

        dom_txt = dom_path.read_text()
        prob_files = sorted(dom_dir.glob("p*.pddl"))
        ns_list=[]; no_list=[]; xs_list=[]; descs=[]; n_tried=0

        for pf in prob_files[:200]:
            if pf.stat().st_size < 50: continue
            n_tried += 1
            try:
                ptxt = pf.read_text()
                ns = run_bfs(dom_txt, ptxt, timeout=8)
                if ns > 0:
                    xs, n_obj = extract_xs(dom_txt, ptxt)
                    ns_list.append(ns); no_list.append(float(n_obj))
                    xs_list.append(xs)
                    descs.append(f"Custom {dom_name} planning with {n_obj} objects. "
                                 f"{dom_data['desc']}")
            except Exception:
                pass
            if n_tried % 50 == 0:
                print(f"  {dom_name}: {n_tried} tried, {len(ns_list)} solved", end="\r")

        print(f"  {dom_name}: {len(ns_list)}/{n_tried} solved            ")
        if len(ns_list) < 8:
            print(f"  SKIP: too few solved"); continue

        ns_arr = np.array(ns_list); no_arr = np.array(no_list)
        Xe_ood = sbert.encode(descs, show_progress_bar=False)
        Xs_n, Xe_n, Xr_n = tfm(np.array(xs_list), Xe_ood)
        arc_sc = []
        with torch.no_grad():
            for i in range(len(Xs_n)):
                qs = torch.FloatTensor(Xs_n[i]).unsqueeze(0)
                qf = torch.FloatTensor(Xe_n[i]).unsqueeze(0)
                qr = torch.FloatTensor(Xr_n[i]).unsqueeze(0)
                try:
                    o, _, _ = model(qs, qf, qr, S_s, S_f, S_V, head="cls")
                    arc_sc.append(float(F.softmax(o.squeeze(0), -1)[1].cpu()))
                except Exception:
                    arc_sc.append(0.5)

        arc_arr = np.array(arc_sc)
        rho_arc, _ = stats.spearmanr(arc_arr, ns_arr)
        rho_obj, _ = stats.spearmanr(no_arr,  ns_arr)
        delta = abs(rho_arc) - abs(rho_obj)
        results[dom_name] = {
            "n": len(ns_list), "bfs_rate": len(ns_list)/n_tried,
            "rho_arc": float(rho_arc), "rho_obj": float(rho_obj),
            "arc_wins": bool(abs(rho_arc) > abs(rho_obj)),
            "desc": dom_data["desc"],
        }
        marker = " ←ARC" if abs(rho_arc) > abs(rho_obj) else ""
        print(f"  {dom_name:<16}  {len(ns_list):>4}  {len(ns_list)/n_tried:>6.1%}  "
              f"{abs(rho_arc):>8.3f}  {abs(rho_obj):>8.3f}  {delta:>+6.3f}{marker}")

    (RES/"custom_ood_evaluation.json").write_text(json.dumps(results, indent=2))
    wins = sum(1 for r in results.values() if r["arc_wins"])
    print(f"\nARC wins: {wins}/{len(results)} custom non-IPC domains")
    print(f"Saved → {RES}/custom_ood_evaluation.json")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--all",      action="store_true")
    args = p.parse_args()
    if args.generate or args.all: generate_all()
    if args.evaluate or args.all: evaluate_all()
    if not any([args.generate, args.evaluate, args.all]):
        print(__doc__)

if __name__ == "__main__":
    main()
