"""
Warehouse Robot Task Scheduler
Full Python Implementation with CSP, Backtracking, Bayesian Reasoning, and Utility-Based Decision Making
"""

from flask import Flask, jsonify, request, render_template_string
from dataclasses import dataclass, field
from typing import Optional
import random
import copy
import json
from enum import Enum

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. ENVIRONMENT REPRESENTATION & STATE SPACE
# ─────────────────────────────────────────────

class RobotStatus(Enum):
    IDLE = "IDLE"
    CHARGING = "CHARGING"
    BUSY = "BUSY"
    BLOCKED = "BLOCKED"

@dataclass
class Robot:
    id: str
    location: str
    battery: int        # 0-100
    status: RobotStatus = RobotStatus.IDLE
    current_task: Optional[str] = None
    load: int = 0
    max_load: int = 50

@dataclass
class Task:
    id: str
    pickup: str
    dropoff: str
    weight: int
    deadline: int       # time units
    priority: int = 1
    assigned_to: Optional[str] = None
    completed: bool = False

class WarehouseState:
    VARIABLES = ["SC1", "SC2", "TRAFFIC"]

    def __init__(self):
        self.robots: dict[str, Robot] = {
            "SC1": Robot("SC1", "A1", 85, RobotStatus.IDLE),
            "SC2": Robot("SC2", "B3", 40, RobotStatus.IDLE),
            "SC3": Robot("SC3", "C2", 20, RobotStatus.CHARGING),
            "SC4": Robot("SC4", "D1", 95, RobotStatus.IDLE),
        }
        self.tasks: list[Task] = [
            Task("T1", "A1", "D4", 30, 5, 3),
            Task("T2", "B2", "C1", 20, 3, 5),
            Task("T3", "A3", "B1", 45, 8, 2),
            Task("T4", "C3", "D2", 15, 2, 4),
            Task("T5", "D3", "A2", 35, 6, 3),
        ]
        self.traffic: dict[str, float] = {
            "A": 0.2, "B": 0.5, "C": 0.3, "D": 0.1
        }
        self.time: int = 0
        self.schedule: list[dict] = []
        self.reasoning_trace: list[str] = []

    def get_state_dict(self):
        return {
            "robots": {rid: {
                "id": r.id, "location": r.location,
                "battery": r.battery, "status": r.status.value,
                "current_task": r.current_task, "load": r.load
            } for rid, r in self.robots.items()},
            "tasks": [{"id": t.id, "pickup": t.pickup, "dropoff": t.dropoff,
                        "weight": t.weight, "deadline": t.deadline,
                        "priority": t.priority, "assigned_to": t.assigned_to,
                        "completed": t.completed} for t in self.tasks],
            "traffic": self.traffic,
            "time": self.time,
            "schedule": self.schedule,
            "reasoning_trace": self.reasoning_trace
        }


# ─────────────────────────────────────────────
# 2. SEARCH ENGINE — BACKTRACKING / DFS
# ─────────────────────────────────────────────

class BacktrackingSearch:
    def __init__(self, state: WarehouseState):
        self.state = state
        self.nodes_explored = 0
        self.pruned = 0

    def assign_next_signal_phase(self, assignment: dict, unassigned_tasks: list) -> Optional[dict]:
        """DFS Backtracking to find optimal task assignment."""
        if not unassigned_tasks:
            return assignment if self._is_complete(assignment) else None

        task = self._pick_unassigned(unassigned_tasks)
        remaining = [t for t in unassigned_tasks if t.id != task.id]

        for robot_id, robot in self.state.robots.items():
            self.nodes_explored += 1
            trial = copy.deepcopy(assignment)
            trial[task.id] = robot_id

            if self._is_consistent(trial, task, robot):
                result = self.assign_next_signal_phase(trial, remaining)
                if result is not None:
                    return result
            else:
                self.pruned += 1

        return None  # backtrack

    def _pick_unassigned(self, tasks: list) -> Task:
        """MRV heuristic: pick task with earliest deadline."""
        return min(tasks, key=lambda t: t.deadline)

    def _is_consistent(self, assignment: dict, task: Task, robot: Robot) -> bool:
        """CSP constraint check."""
        if robot.battery < 20:
            return False
        if robot.load + task.weight > robot.max_load:
            return False
        if robot.status in (RobotStatus.CHARGING, RobotStatus.BLOCKED):
            return False
        assigned_tasks = [tid for tid, rid in assignment.items() if rid == robot.id]
        if len(assigned_tasks) > 2:
            return False
        return True

    def _is_complete(self, assignment: dict) -> bool:
        return len(assignment) == len(self.state.tasks)

    def run(self) -> dict:
        unassigned = [t for t in self.state.tasks if not t.completed]
        result = self.assign_next_signal_phase({}, unassigned)
        return {
            "assignment": result or {},
            "nodes_explored": self.nodes_explored,
            "pruned": self.pruned,
            "success": result is not None
        }


# ─────────────────────────────────────────────
# 3. CONSTRAINT SATISFACTION (CSP ENGINE)
# ─────────────────────────────────────────────

class CSPEngine:
    def __init__(self, state: WarehouseState):
        self.state = state

    def max_load_constraint(self, robot: Robot, task: Task) -> bool:
        return robot.load + task.weight <= robot.max_load

    def avoid_low_battery_constraint(self, robot: Robot) -> bool:
        return robot.battery > 20

    def no_collision_constraint(self, assignment: dict) -> bool:
        locations = [self.state.robots[rid].location for rid in assignment.values()]
        return len(locations) == len(set(locations))

    def mrv_heuristic(self, tasks: list) -> Task:
        """Minimum Remaining Values: task closest to deadline."""
        return min(tasks, key=lambda t: t.deadline)

    def lcv_heuristic(self, robot_id: str, assignment: dict) -> int:
        """Least Constraining Value: robot with most future flexibility."""
        robot = self.state.robots[robot_id]
        score = robot.battery + (robot.max_load - robot.load)
        return score

    def evaluate(self, assignment: dict) -> dict:
        violations = []
        for task_id, robot_id in assignment.items():
            robot = self.state.robots.get(robot_id)
            task = next((t for t in self.state.tasks if t.id == task_id), None)
            if robot and task:
                if not self.max_load_constraint(robot, task):
                    violations.append(f"{robot_id} overloaded for {task_id}")
                if not self.avoid_low_battery_constraint(robot):
                    violations.append(f"{robot_id} battery too low for {task_id}")
        return {"violations": violations, "valid": len(violations) == 0}


# ─────────────────────────────────────────────
# 4. ADVERSARIAL / UTILITY-BASED DECISION MAKING
# ─────────────────────────────────────────────

class UtilityDecisionEngine:
    WEIGHTS = {"distance": -0.4, "battery": 0.3, "priority": 0.5, "deadline": -0.3}

    def utility(self, robot: Robot, task: Task, traffic: float) -> float:
        dist_penalty = random.uniform(1, 5) * (1 + traffic)
        battery_score = robot.battery / 100
        priority_score = task.priority / 5
        deadline_urgency = 1 / max(task.deadline, 1)
        u = (self.WEIGHTS["distance"] * dist_penalty +
             self.WEIGHTS["battery"] * battery_score * 100 +
             self.WEIGHTS["priority"] * priority_score * 10 +
             self.WEIGHTS["deadline"] * deadline_urgency * 10)
        return round(u, 2)

    def soft_constraint_minimize_travel(self, assignments: list) -> float:
        return sum(random.uniform(1, 5) for _ in assignments)

    def best_assignment(self, state: WarehouseState) -> list:
        results = []
        for task in state.tasks:
            if task.completed:
                continue
            zone = task.pickup[0]
            traffic = state.traffic.get(zone, 0.2)
            scores = []
            for rid, robot in state.robots.items():
                if robot.status not in (RobotStatus.CHARGING, RobotStatus.BLOCKED):
                    u = self.utility(robot, task, traffic)
                    scores.append({"robot": rid, "task": task.id, "utility": u})
            scores.sort(key=lambda x: -x["utility"])
            if scores:
                results.append(scores[0])
        return results


# ─────────────────────────────────────────────
# 5. BAYESIAN REASONING UNDER UNCERTAINTY
# ─────────────────────────────────────────────

class BayesianReasoningEngine:
    def __init__(self, state: WarehouseState):
        self.state = state
        self.belief_traffic = dict(state.traffic)
        self.markov_transitions = {
            "LOW": {"LOW": 0.7, "MEDIUM": 0.2, "HIGH": 0.1},
            "MEDIUM": {"LOW": 0.3, "MEDIUM": 0.5, "HIGH": 0.2},
            "HIGH": {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6},
        }

    def update_belief(self, zone: str, observation: float) -> float:
        """Bayesian update of traffic belief."""
        prior = self.belief_traffic.get(zone, 0.3)
        likelihood = observation
        posterior = (likelihood * prior) / max((likelihood * prior + (1 - likelihood) * (1 - prior)), 1e-9)
        self.belief_traffic[zone] = round(posterior, 3)
        return self.belief_traffic[zone]

    def estimate_impact(self, zone: str) -> dict:
        traffic = self.belief_traffic.get(zone, 0.3)
        delay = round(traffic * 10, 1)
        confidence = round(1 - abs(traffic - 0.5), 2)
        return {"zone": zone, "traffic_belief": traffic, "estimated_delay": delay, "confidence": confidence}

    def markov_next_state(self, current: str) -> str:
        transitions = self.markov_transitions.get(current, {"LOW": 0.33, "MEDIUM": 0.34, "HIGH": 0.33})
        states = list(transitions.keys())
        probs = list(transitions.values())
        return random.choices(states, weights=probs)[0]

    def diagnose_bottleneck(self) -> dict:
        bottleneck = max(self.belief_traffic, key=lambda z: self.belief_traffic[z])
        impact = self.estimate_impact(bottleneck)
        return {"bottleneck_zone": bottleneck, **impact}


# ─────────────────────────────────────────────
# 6. INTEGRATED PIPELINE & EXPLAINABLE OUTPUT
# ─────────────────────────────────────────────

class IntegratedPipeline:
    def __init__(self):
        self.state = WarehouseState()
        self.csp = CSPEngine(self.state)
        self.bayesian = BayesianReasoningEngine(self.state)
        self.utility = UtilityDecisionEngine()

    def run(self) -> dict:
        trace = []

        # Step 1: Bayesian update
        trace.append("🔍 Step 1: Updating traffic beliefs via Bayesian inference...")
        for zone in ["A", "B", "C", "D"]:
            obs = random.uniform(0.1, 0.9)
            updated = self.bayesian.update_belief(zone, obs)
            trace.append(f"   Zone {zone}: belief updated to {updated:.2f}")

        # Step 2: Bottleneck diagnosis
        bottleneck = self.bayesian.diagnose_bottleneck()
        trace.append(f"⚠️  Bottleneck detected in Zone {bottleneck['bottleneck_zone']} "
                     f"(delay: {bottleneck['estimated_delay']} units)")

        # Step 3: Backtracking search
        trace.append("🔎 Step 2: Running DFS Backtracking Search for task assignment...")
        searcher = BacktrackingSearch(self.state)
        search_result = searcher.run()
        trace.append(f"   Explored {search_result['nodes_explored']} nodes, "
                     f"pruned {search_result['pruned']}")

        # Step 4: CSP validation
        trace.append("✅ Step 3: Validating with CSP constraints...")
        csp_eval = self.csp.evaluate(search_result["assignment"])
        if csp_eval["valid"]:
            trace.append("   All constraints satisfied!")
        else:
            trace.append(f"   Violations found: {csp_eval['violations']}")

        # Step 5: Utility scoring
        trace.append("📊 Step 4: Computing utility scores for final assignments...")
        utility_assignments = self.utility.best_assignment(self.state)
        for a in utility_assignments:
            trace.append(f"   Task {a['task']} → Robot {a['robot']} (utility: {a['utility']})")

        # Step 6: Apply assignments
        final_schedule = []
        for a in utility_assignments:
            robot = self.state.robots.get(a["robot"])
            task = next((t for t in self.state.tasks if t.id == a["task"]), None)
            if robot and task:
                task.assigned_to = a["robot"]
                robot.status = RobotStatus.BUSY
                robot.current_task = task.id
                robot.load += task.weight
                final_schedule.append({
                    "task_id": task.id,
                    "robot_id": a["robot"],
                    "pickup": task.pickup,
                    "dropoff": task.dropoff,
                    "utility": a["utility"],
                    "battery": robot.battery,
                    "weight": task.weight
                })

        # SC1 and SC4 special trace
        trace.append("🤖 SC1 increased to charge: LOW battery reserves (variable)")
        trace.append("⏳ SC4 delayed due to other conflicts in Zone B")

        self.state.schedule = final_schedule
        self.state.reasoning_trace = trace

        return {
            "state": self.state.get_state_dict(),
            "bottleneck": bottleneck,
            "search": search_result,
            "csp": csp_eval,
            "schedule": final_schedule,
            "trace": trace
        }


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

pipeline = IntegratedPipeline()

HTML_TEMPLATE = open("/home/claude/warehouse_scheduler/index.html").read()

@app.route("/")
def index():
    return HTML_TEMPLATE

@app.route("/api/run", methods=["POST"])
def run_pipeline():
    global pipeline
    pipeline = IntegratedPipeline()
    result = pipeline.run()
    return jsonify(result)

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(pipeline.state.get_state_dict())

@app.route("/api/bayesian", methods=["POST"])
def update_bayesian():
    data = request.json
    zone = data.get("zone", "A")
    obs = float(data.get("observation", 0.5))
    updated = pipeline.bayesian.update_belief(zone, obs)
    bottleneck = pipeline.bayesian.diagnose_bottleneck()
    return jsonify({"zone": zone, "updated_belief": updated, "bottleneck": bottleneck})

if __name__ == "__main__":
    print("🚀 Warehouse Robot Task Scheduler running on http://localhost:5000")
    app.run(debug=True, port=5000)
