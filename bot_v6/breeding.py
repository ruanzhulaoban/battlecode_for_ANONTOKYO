"""Food-sensitive breeding with a bounded, uncalibrated collision-risk model."""
from math import prod

# Active breeding leaves these team slots for emergency splits. Emergency
# splits may consume the reserve; it is not an exemption from the engine cap.
EMERGENCY_SLOTS = 4


class Breeding:
    def __init__(self):
        self.body = []  # Confirmed prefix, head first; not a guessed straight tail.
        self.previous = None
        self.next_heads = set()
        self.memory = {}  # Position -> (turn, destinations, occupant id, enemy, head).
        self.tail = None
        self.tier = 'poor'
        self.supply = 0.0
        self.cooldown = 8
        self.risk_limit = .15
        self.risk = 1.0
        self.reason = 'not evaluated'

    def observe(self, p, ct, game, directions):
        turn = game.get_round_num()
        head = ct.get_position()
        origin = (head.x, head.y)
        length = ct.get_length()
        own = {q: part for q, part in p.occupied.items() if part.get_id() == p.me}
        predecessors = {}
        for q, part in own.items():
            if not part.is_head():
                predecessors.setdefault(p.destination(q, part.get_dir()), []).append(q)
        visible = [origin]
        while len(visible) < length:
            choices = predecessors.get(visible[-1], [])
            if len(choices) != 1 or choices[0] in visible:
                break
            visible.append(choices[0])
        candidate = []
        if self.previous is not None:
            old_turn, old_head, old_length = self.previous
            if turn == old_turn + 1:
                if origin != old_head and origin in self.next_heads and length in (old_length, old_length + 1):
                    candidate = ([origin] + self.body)[:length]
                elif (origin == old_head and
                      (length == old_length or 2 <= length <= old_length - 2)):
                    # Either end may now retain most of the length after a split.
                    candidate = self.body[:length]
            elif turn == old_turn and origin == old_head and length == old_length:
                candidate = self.body[:]
        # Reject history contradicted by CURRENT visibility, including missing
        # body cells and newly observed own cells outside a supposedly full chain.
        valid = (candidate and len(set(candidate)) == len(candidate)
                 and candidate[:min(len(candidate), len(visible))] == visible[:min(len(candidate), len(visible))]
                 and all(q not in p.tiles or q in own for q in candidate)
                 and (len(candidate) < length or set(own).issubset(candidate)))
        self.body = candidate if valid and len(candidate) > len(visible) else visible
        self.tail = self.body[-1] if len(self.body) == length else None
        self.previous = (turn, origin, length)
        self.next_heads = {q for q in p.graph[origin] if q is not None and q not in p.occupied}
        for q, tile in p.tiles.items():
            destinations = []
            for direction in directions:
                kind = tile.get_edge(direction).get_edge_type().name
                if kind == 'EMPTY':
                    dx, dy = direction.get_offset()
                    dest = ((q[0] + dx) % p.width, (q[1] + dy) % p.height)
                elif kind == 'PORTAL':
                    dest = p.destination(q, direction)  # Unknown portal exits stay unknown.
                else:
                    dest = None
                destinations.append(dest)
            part = p.occupied.get(q)
            self.memory.pop(q, None)
            self.memory[q] = (turn, tuple(destinations),
                              part.get_id() if part else None,
                              part is not None and part.get_team() != p.team,
                              part is not None and part.is_head())
        while len(self.memory) > 512:
            del self.memory[next(iter(self.memory))]
        self.assess_food(p, origin)

    def assess_food(self, p, origin):
        reachable = p.space(origin, set(p.occupied))
        nearby = {q for q, distance in reachable.items() if distance <= 6}
        available = len(p.pearls.intersection(nearby))
        expected = float(available)
        for q in nearby - p.pearls:
            delay = p.tiles[q].get_pearl_time()
            if 0 <= delay <= 4:
                expected += .5
            elif 4 < delay <= 8:
                expected += .25
        competitors = {part.get_id() for q, part in p.occupied.items()
                       if part.get_team() == p.team and part.get_id() != p.me
                       and p.sonar.distance(p, origin, q) <= 4}
        self.supply = expected / (1 + len(competitors))
        density = expected / max(1, len(nearby))
        self.tier = ('rich' if available >= 2 and self.supply >= 2 and density >= .06
                     else 'normal' if self.supply >= .5 else 'poor')
        self.cooldown = {'rich': 2, 'normal': 4, 'poor': 8}[self.tier]
        self.risk_limit = {'rich': .35, 'normal': .25, 'poor': .15}[self.tier]

    def policy(self, p, ct, game):
        turn = game.get_round_num()
        open_area = p.local_open >= .80 and p.open_score >= .85
        if turn <= 120:
            area = 24 if open_area else 48
        elif turn <= 300:
            area = 32 if open_area else 64
        else:
            # Lower late-game demand, but still replace losses after round 450.
            area = 48 if open_area else 80
        # Length only enforces the engine's two-cells-per-snake minimum.
        # Food controls population demand, cooldown and tolerated conflict risk,
        # rather than requiring scarce-food snakes to grow before expanding.
        area *= {'rich': .65, 'normal': 1, 'poor': 3}[self.tier]
        limit = game.get_unit_limit()
        reserve = min(EMERGENCY_SLOTS, max(0, limit - 2))
        target = min(limit - reserve, max(2, int(p.width * p.height / area)))
        return 4, target

    def split_size(self, p, ct, game, origin, blocked, emergency=False):
        """Give length to the more escapable end using at most two local steps.

        Called only when already committed to splitting. No BFS, flood fill or
        recursive search: at most 4 exits and 4 continuations per end. Recent
        history is weaker evidence than current visibility, never proof of safety.
        """
        length = ct.get_length()
        if length <= 4:
            return 2
        turn = game.get_round_num()
        occupied = blocked | set(self.body)
        # One-step head collision hints; do not invoke the full threat search.
        danger = {q for head, _ in p.heads for q in p.graph.get(head, ())
                  if q is not None}

        def neighbours(q):
            if q in p.tiles:
                return p.graph.get(q, ())
            record = self.memory.get(q)
            if record is not None and 0 <= turn - record[0] <= 3:
                return record[1]
            return ()

        def clearance(q):
            if q is None or q in occupied:
                return 0
            if q in p.tiles:
                return 1 if q in danger else 3
            record = self.memory.get(q)
            if (record is not None and 0 <= turn - record[0] <= 3
                    and record[2] in (None, p.me)):
                return 2
            return 0

        def escape_score(head):
            routes = visible = historic_routes = historic = risky = 0
            for q in set(neighbours(head)) - {None}:
                quality = clearance(q)
                if not quality:
                    continue
                onward = max((clearance(dest) for dest in neighbours(q)
                              if dest != head), default=0)
                if quality == 3:
                    visible += 1
                    routes += onward == 3
                elif quality == 2:
                    historic += 1
                    historic_routes += onward >= 2
                else:
                    risky += 1
            return routes, visible, historic_routes, historic, risky

        front = escape_score(origin)
        tail = self.tail if self.tail is not None else p.tail
        back = escape_score(tail) if tail is not None else (0, 0, 0, 0, 0)
        if back > front:
            return length - 2
        if emergency and not any(front) and not any(back):
            # The front has no verified move. If the other end is unobserved,
            # put the long body there so its new head can inspect its own exits.
            # A currently observed blocked tail offers no such alternative.
            record = self.memory.get(tail)
            remembered_walls = (record is not None and
                                0 <= turn - record[0] <= 3 and
                                not any(q is not None for q in record[1]))
            if tail not in p.tiles and not remembered_walls:
                return length - 2
        # Equal evidence or a better front: retain length at the original head.
        return 2

    def estimate_risk(self, p, ct, game, origin, blocked):
        """Heuristic next-action conflict probability, not a calibrated rate.

        Current observations veto known threats; historical cells contribute
        staleness, occupancy and nearby-head risk. Product assumes independent
        exit failures, with an extra shared uncertainty term to temper optimism.
        """
        self.risk = 1.0
        self.reason = 'unknown tail'
        if self.tail is None:
            return self.risk
        turn = game.get_round_num()
        blocked = blocked | set(self.body)
        escape_sets = []
        for head in (origin, self.tail):
            if head not in p.tiles:
                escape_sets.append(None)
                continue
            direct, sprint, danger = p.threats(head, blocked)
            if direct or sprint:
                self.reason = 'visible head threat'
                return self.risk
            exits = {q for q in p.graph[head] if q is not None and q not in blocked and q not in danger}
            if not exits or len(p.space(head, blocked | danger)) < max(12, ct.get_length() + 4):
                self.reason = 'insufficient visible escape space'
                return self.risk
            escape_sets.append(exits)
        if escape_sets[1] is not None:
            if len(escape_sets[0] | escape_sets[1]) < 2:
                self.reason = 'shared only exit'
                return self.risk
            self.risk, self.reason = .02, 'observed tail'
            return self.risk

        record = self.memory.get(self.tail)
        if record is None or not 0 <= turn - record[0] <= 8:
            self.reason = 'tail neighbourhood too old'
            return self.risk
        age = turn - record[0]
        foreign_density = sum(part.get_id() != p.me for part in p.occupied.values()) / max(1, len(p.tiles))
        exit_risks = []
        for q in set(record[1]) - {None}:
            if q in blocked:
                continue
            if q in p.tiles:
                direct, sprint, _ = p.threats(q, blocked)
                risk = .8 if direct else .5 if sprint else .03
            else:
                previous = self.memory.get(q)
                risk = min(.85, .12 + .04 * age + foreign_density)
                if previous is None:
                    risk = max(risk, .65)
                elif previous[2] is not None and previous[2] != p.me:
                    risk = max(risk, .8 if previous[3] else .6)
            exit_risks.append(risk)
        if not exit_risks:
            self.reason = 'no remembered tail exit'
            return self.risk
        shared = min(.7, .04 + .025 * age + .5 * foreign_density + .25 * (1 - p.local_open))
        threat = .15 if turn - p.sonar.enemy_echo_turn <= 4 else 0.0
        for q, part in p.heads:
            if p.sonar.distance(p, q, self.tail) <= 2:
                threat = max(threat, .7 if part.get_team() != p.team else .4)
        for q, old in self.memory.items():
            elapsed = turn - old[0]
            if (q not in p.tiles and old[4] and old[2] != p.me and 0 <= elapsed <= 4
                    and p.sonar.distance(p, q, self.tail) <= elapsed + 2):
                threat = max(threat, .65 if old[3] else .35)
        # Historical occupied cells raise exit risk; no claim of empty space.
        self.risk = min(1.0, 1 - (1 - shared) * (1 - prod(exit_risks)) * (1 - threat))
        self.reason = f'history age={age} exits={len(exit_risks)}'
        return self.risk
