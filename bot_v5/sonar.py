"""Bounded, lossy sonar hints. Packets claim identity; they do not prove it."""
from typing import NamedTuple

PROBE, FOOD, CROWD, SPACE, ENEMY = range(5)
TTL = {PROBE: 0, FOOD: 16, CROWD: 8, SPACE: 4, ENEMY: 8}
MAGIC = 0xD4


class Report(NamedTuple):
    kind: int
    team: int
    turn: int
    source: int
    x: int
    y: int
    value: int
    hop: int = 0


def encode(report):
    """8-bit marker + 3/1/9/12/9/9/12/1-bit fields = 64 bits."""
    word = MAGIC
    for value, bits in zip(report, (3, 1, 9, 12, 9, 9, 12, 1)):
        if type(value) is not int or not 0 <= value < 1 << bits:
            return None
        word = (word << bits) | value
    return word


def decode(word, team, turn, width, height):
    if type(word) is not int or not 0 <= word < 1 << 64:
        return None
    values = []
    for bits in (1, 12, 9, 9, 12, 9, 1, 3):
        values.append(word & ((1 << bits) - 1))
        word >>= bits
    report = Report(*reversed(values))
    if (word != MAGIC or report.kind not in TTL or report.team != team
            or not 0 <= turn - report.turn <= TTL[report.kind]
            or report.x >= width or report.y >= height
            or (report.kind in (SPACE, CROWD) and report.hop)):
        return None
    return report


class Sonar:
    def __init__(self):
        self.reports = {}
        self.last_probe = None
        self.risks = {}  # Direction -> (strength, expiry, emission position).
        self.enemy_echo_turn = -100
        self.jam_turns = 0
        self.relayed = {}  # Packet -> original observation turn, not receipt time.
        self.request = None
        self.goal = None
        self.avoid = []
        self.crowded_scarce = False
        self.local_food = set()

    @staticmethod
    def distance(p, a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return min(dx, p.width - dx) + min(dy, p.height - dy)

    def receive(self, p, ct, turn, origin):
        team = int(p.team.value == 'B')
        self.reports = {k: r for k, r in self.reports.items()
                        if 0 <= turn - r.turn <= TTL[r.kind]}
        self.relayed = {k: t for k, t in self.relayed.items() if turn - t <= 16}
        self.risks = {d: entry for d, entry in self.risks.items()
                      if entry[1] >= turn and self.distance(p, origin, entry[2]) <= 3}
        echoes = ct.get_sonar_echoes()
        if self.last_probe is not None and self.last_probe[0] == turn - 1:
            _, direction, position = self.last_probe
            if echoes.enemy or echoes.enemy_head:
                self.enemy_echo_turn = turn
                if direction is not None:
                    strength = min(3, self.risks.get(direction, (0, 0, None))[0] + 1)
                    self.risks[direction] = (strength, turn + 6, position)
            elif direction is not None:
                self.risks.pop(direction, None)
        # Fixed processing budget even if many enemies spam our inbox.
        for word in ct.get_sonar_messages()[:128]:
            r = decode(word, team, turn, p.width, p.height)
            if r is None or r.kind == PROBE or r.source == p.me:
                continue
            # Reject identities contradicted by our own observation.
            if any(part.get_id() == r.source and part.get_team() != p.team
                   for part in p.occupied.values()):
                continue
            key = (r.kind, r.value if r.kind == ENEMY else r.source)
            old = self.reports.get(key)
            if old is None or r.turn > old.turn:
                self.reports[key] = r
        # Directly observed depletion overrides a remote food claim.
        self.reports = {k: r for k, r in self.reports.items()
                        if not (r.kind == FOOD and (r.x, r.y) in p.tiles
                                and (r.x, r.y) not in p.pearls)}
        if len(self.reports) > 64:
            self.reports = dict(sorted(self.reports.items(),
                                       key=lambda item: item[1].turn)[-64:])

    def prepare(self, p, ct, turn, origin, blocked):
        self.local_food = p.pearls.intersection(p.space(origin, blocked))
        friendly = {}
        for part in p.occupied.values():
            if part.get_team() == p.team:
                ident = part.get_id()
                friendly[ident] = friendly.get(ident, 0) + 1
        crowded = sum(friendly.values()) * 3 >= len(p.tiles)
        self.crowded_scarce = crowded and len(self.local_food) <= 1
        exits = sum(q is not None and q not in blocked for q in p.graph[origin])
        self.jam_turns = self.jam_turns + 1 if crowded and exits <= 1 else 0
        self.request = None
        requests = [r for r in self.reports.values()
                    if r.kind == SPACE and r.value > ct.get_length()
                    and friendly.get(r.source, 0) > ct.get_length()
                    and self.distance(p, origin, (r.x, r.y)) <= 4]
        if requests:
            self.request = max(requests, key=lambda r: (r.value, r.turn, -r.source))
        # Only a subset follows a food report; others continue exploring.
        foods = [r for r in self.reports.values() if r.kind == FOOD
                 and (p.me + r.source + r.x + r.y) % 3 == 0]
        self.avoid = [r for r in self.reports.values()
                      if r.kind in (CROWD, ENEMY) and
                      self.distance(p, origin, (r.x, r.y)) <= 6]
        foods = [r for r in foods if not any(
            self.distance(p, (r.x, r.y), (a.x, a.y)) <= 3 for a in self.avoid)]
        self.goal = min(foods, key=lambda r: (
            self.distance(p, origin, (r.x, r.y)) + turn - r.turn - min(r.value, 4),
            (r.source + p.me) % 4096), default=None)

    def may_retire(self, p, ct, turn):
        if self.request is None or turn - self.enemy_echo_turn <= 4:
            return False
        if turn % 4 != p.me % 4:
            return False
        # Among visible plausible short responders, only the smallest ID acts.
        # Partial bodies are lower bounds: including long snakes here is safe.
        counts = {}
        for part in p.occupied.values():
            if part.get_team() == p.team:
                ident = part.get_id()
                counts[ident] = counts.get(ident, 0) + 1
        candidates = [p.me] + [part.get_id() for _, part in p.heads
                               if part.get_team() == p.team
                               and counts[part.get_id()] <= 4
                               and part.get_id() != self.request.source]
        return p.me == min(candidates)

    def movement_hint(self, p, dest, distances, direction):
        score = -3 * self.risks.get(direction, (0, 0, None))[0]
        if self.request is not None:
            score += 2 * min(6, self.distance(p, dest, (self.request.x, self.request.y)))
        for r in self.avoid:
            score -= max(0, 4 - self.distance(p, dest, (r.x, r.y)))
        if self.goal is not None:
            target = (self.goal.x, self.goal.y)
            # Route to a useful visible waypoint, never through unseen occupancy.
            travel = min(n + self.distance(p, q, target) for q, n in distances.items())
            score -= travel + 2 * p.visits.get(dest, 0)
        return score

    def emit(self, p, ct, turn, origin, after_head, directions):
        team = int(p.team.value == 'B')
        def report(kind, pos, value):
            return Report(kind, team, turn, p.me, *pos, min(value, 4095))

        outgoing = None
        if self.jam_turns >= 3 and ct.get_length() >= 5:
            outgoing = report(SPACE, origin, ct.get_length())
        elif self.crowded_scarce:
            outgoing = report(CROWD, origin, ct.get_length())
        elif turn % 2 == 0 and any(part.get_team() != p.team for _, part in p.heads):
            pos, part = next((pos, part) for pos, part in p.heads if part.get_team() != p.team)
            if part.get_id() < 4096:
                outgoing = report(ENEMY, pos, part.get_id())
        elif len(self.local_food) >= 2:
            pos = min(self.local_food, key=lambda q: (self.distance(p, origin, q), q))
            outgoing = report(FOOD, pos, len(self.local_food))
        if outgoing is None:
            eligible = [r for r in self.reports.values()
                        if r.kind in (FOOD, ENEMY) and not r.hop
                        and r not in self.relayed]
            if eligible:
                r = max(eligible, key=lambda r: r.turn)
                self.relayed[r] = r.turn
                if len(self.relayed) > 128:
                    del self.relayed[next(iter(self.relayed))]
                outgoing = r._replace(hop=1)
        if outgoing is None:
            outgoing = report(PROBE, origin, 0)
        # One beam gives the aggregate echo an attributable launch direction.
        direction = directions[(turn + p.me) % 4]
        word = encode(outgoing)
        self.last_probe = None
        if word is not None and ct.send_sonar(direction, word):
            # Visible portals or our own body make bearing interpretation unsafe.
            pos, checked = after_head, set()
            bearing = direction
            while pos in p.tiles and pos not in checked:
                checked.add(pos)
                if p.tiles[pos].get_edge(direction).get_edge_type().name == 'PORTAL':
                    bearing = None
                    break
                q = p.destination(pos, direction)
                if q is None:
                    break
                if q in p.occupied:
                    if p.occupied[q].get_id() == p.me:
                        bearing = None
                    break
                pos = q
            self.last_probe = (turn, bearing, after_head)
