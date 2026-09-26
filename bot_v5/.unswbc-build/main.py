"""Food-sensitive breeding, remembered tails and bounded sonar cooperation."""
from collections import deque
if __package__:
    from . import helper as bc
    from .sonar import Sonar
    from .breeding import Breeding
else:
    import helper as bc
    from sonar import Sonar
    from breeding import Breeding

DIRS = bc.Direction.get_direction_list()


class Planner:
    def __init__(self):
        self.sonar = Sonar()
        self.breeding = Breeding()
        self.visits = {}
        self.portals = {}
        self.open_score = None
        self.last_split = -100
        self.explore_dir = None
        self.explore_until = -1
        self.seen = {}  # Bounded, recent terrain observations, never occupancy.
        self.last_meal = None
        self.previous_length = None
        self.pressure_turns = 0

    def expansion_targets(self, turn):
        """Visible frontier cells; movement still requires a verified destination."""
        self.seen.update((pos, turn) for pos in self.tiles)
        if len(self.seen) > 2048:
            self.seen = dict(sorted(self.seen.items(), key=lambda item: item[1])[-2048:])
        frontier = set()
        for pos, tile in self.tiles.items():
            for direction in DIRS:
                if tile.get_edge(direction).get_edge_type() != bc.EdgeType.EMPTY:
                    continue
                dx, dy = direction.get_offset()
                q = ((pos[0] + dx) % self.width, (pos[1] + dy) % self.height)
                if q not in self.tiles and turn - self.seen.get(q, -1000) >= 32:
                    frontier.add(pos)
                    break
        # On fully explored/visible maps, seek cells our head has not visited.
        return frontier or {p for p in self.tiles if not self.visits.get(p, 0)}

    def population_pressure(self, ct, origin, blocked):
        """Local dominance is only a proxy: the protocol has no enemy census."""
        allies, enemies = {}, set()
        for part in self.occupied.values():
            ident = part.get_id()
            if part.get_team() == self.team:
                allies[ident] = allies.get(ident, 0) + 1
            else:
                enemies.add(ident)
        reachable = self.space(origin, blocked)
        crowded = sum(allies.values()) * 3 >= len(self.tiles)
        scarce = len(self.pearls.intersection(reachable)) <= 1
        dominant = (ct.get_unit_count() >= 8
                    and len(allies) >= max(4, 3 * len(enemies)))
        pressured = crowded and scarce and dominant
        self.pressure_turns = self.pressure_turns + 1 if pressured else 0
        # Visible segment counts are lower bounds; only a demonstrably longer
        # neighbour warrants giving up this short snake's emergency rescue.
        retire = (self.pressure_turns >= 3 and ct.get_length() <= 4
                  and any(n > ct.get_length() for ident, n in allies.items()
                          if ident != self.me))
        return pressured, retire

    def snapshot(self, ct, game):
        self.width, self.height = game.get_map_size()
        self.me = ct.get_id()
        self.team = ct.get_team()
        self.tiles = {}
        self.occupied = {}
        self.heads = []
        self.pearls = set()
        kelp_edges = 0
        for tile in ct.get_tiles():
            p = tile.get_position()
            pos = (p.x, p.y)
            self.tiles[pos] = tile
            if tile.has_pearl():
                self.pearls.add(pos)
            part = tile.get_dragon()
            if part is not None:
                self.occupied[pos] = part
                if part.is_head() and part.get_id() != self.me:
                    self.heads.append((pos, part))
            for d in DIRS:
                edge = tile.get_edge(d)
                kelp_edges += edge.get_edge_type() == bc.EdgeType.KELP
                if edge.get_edge_type() == bc.EdgeType.PORTAL:
                    key = self.edge_key(pos, d)
                    self.portals.setdefault(edge.get_portal_id(), set()).add(key)
        # Local structural estimate, not a claim to know the unseen map.
        self.local_open = 1 - kelp_edges / max(1, 4 * len(self.tiles))
        self.open_score = (self.local_open if self.open_score is None else
                           (3 * self.open_score + self.local_open) / 4)
        self.graph = {}
        for pos in self.tiles:
            self.graph[pos] = [self.destination(pos, d) for d in DIRS]
        # A fully visible, unambiguous body can be reconstructed head to tail.
        own = {p: part for p, part in self.occupied.items()
               if part.get_id() == self.me}
        self.tail = None
        if len(own) == ct.get_length():
            predecessors = {}
            for pos, part in own.items():
                if not part.is_head():
                    toward_head = self.destination(pos, part.get_dir())
                    predecessors.setdefault(toward_head, []).append(pos)
            head = ct.get_position()
            chain = [(head.x, head.y)]
            while len(chain) < len(own):
                choices = predecessors.get(chain[-1], [])
                if len(choices) != 1 or choices[0] in chain:
                    break
                chain.append(choices[0])
            if len(chain) == len(own):
                self.tail = chain[-1]
        self.breeding.observe(self, ct, game, DIRS)

    def reproduction_policy(self, ct, game):
        return self.breeding.policy(self, ct, game)

    def try_reproduce(self, ct, game, origin, blocked):
        minimum, target = self.reproduction_policy(ct, game)
        if (not minimum or ct.get_length() < minimum
                or ct.get_unit_count() >= target or not ct.can_split(2)
                or game.get_round_num() - self.last_split < self.breeding.cooldown
                or len(self.tiles) < 16
                or len(blocked) * 4 > len(self.tiles)):
            return False
        risk = self.breeding.estimate_risk(self, ct, game, origin, blocked)
        if risk > self.breeding.risk_limit:
            return False
        child_size = self.breeding.split_size(self, ct, game, origin, blocked)
        self.last_split = game.get_round_num()
        ct.set_indicator_string(f'v5 breed food={self.breeding.tier} supply={self.breeding.supply:.2f} '
                                f'risk_est={risk:.2f} tail={self.breeding.reason} '
                                f'units={ct.get_unit_count()}/{target} '
                                f'front={ct.get_length() - child_size} back={child_size}')
        ct.do_split(child_size)
        return True

    def edge_key(self, pos, direction):
        x, y = pos
        if direction == bc.Direction.NORTH:
            return ('h', x, y)
        if direction == bc.Direction.SOUTH:
            return ('h', x, (y + 1) % self.height)
        if direction == bc.Direction.WEST:
            return ('v', x, y)
        return ('v', (x + 1) % self.width, y)

    def destination(self, pos, direction):
        tile = self.tiles.get(pos)
        if tile is None:
            return None
        edge = tile.get_edge(direction)
        kind = edge.get_edge_type()
        if kind == bc.EdgeType.KELP:
            return None
        if kind == bc.EdgeType.PORTAL:
            here = self.edge_key(pos, direction)
            pair = self.portals.get(edge.get_portal_id(), set())
            if len(pair) != 2 or here not in pair:
                return None
            _, x, y = next(p for p in pair if p != here)
            if direction == bc.Direction.NORTH:
                y = (y - 1) % self.height
            elif direction == bc.Direction.WEST:
                x = (x - 1) % self.width
            dest = (x, y)
        else:
            dx, dy = direction.get_offset()
            dest = ((pos[0] + dx) % self.width, (pos[1] + dy) % self.height)
        # Static portal knowledge never certifies unseen dynamic occupancy.
        return dest if dest in self.tiles else None

    def space(self, start, blocked):
        """BFS doubles as flood fill and pearl-distance evaluation."""
        distance = {start: 0}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            for dest in self.graph.get(pos, ()):
                if dest is not None and dest not in blocked and dest not in distance:
                    distance[dest] = distance[pos] + 1
                    queue.append(dest)
        return distance

    def threats(self, target, blocked):
        """Possible one/two-step head attacks; not probability estimates.

        Other dragons' lengths/tails are not fully observed. Two-step paths
        therefore conservatively assume the attacker can pay for sprinting.
        Occupied cells are fixed except the victim head, which terminates a ray.
        """
        direct = sprint = 0
        endpoints = set()
        for origin, part in self.heads:
            enemy = part.get_team() != self.team
            # Allies are also collision hazards, with no shared-memory promise.
            weight = 2 if enemy else 1
            hit1 = hit2 = False
            for first in self.graph.get(origin, ()):
                if first is None:
                    continue
                if first == target:
                    hit1 = True
                    continue
                if first in blocked:
                    continue
                endpoints.add(first)
                for second in self.graph.get(first, ()):
                    if second is None or second == first or second == origin:
                        continue
                    if second == target:
                        hit2 = True
                    elif second not in blocked:
                        endpoints.add(second)
            direct += weight * hit1
            sprint += weight * hit2
        return direct, sprint, endpoints

    def choose(self, ct, game):
        self.snapshot(ct, game)
        p = ct.get_position()
        origin = (p.x, p.y)
        self.visits[origin] = self.visits.get(origin, 0) + 1
        blocked = set(self.occupied)
        turn = game.get_round_num()
        if (self.last_meal is None or (self.previous_length is not None
                                     and ct.get_length() > self.previous_length)):
            self.last_meal = turn
        self.previous_length = ct.get_length()
        self.sonar.receive(self, ct, turn, origin)
        targets = self.expansion_targets(turn)
        _, retire = self.population_pressure(ct, origin, blocked)
        self.sonar.prepare(self, ct, turn, origin, blocked)
        retire = retire and self.sonar.may_retire(self, ct, turn)
        if (self.sonar.request is None
                and self.try_reproduce(ct, game, origin, blocked)):
            self.sonar.emit(self, ct, turn, origin, origin, DIRS)
            return
        heading = ct.get_dir()
        if self.explore_dir is None or turn >= self.explore_until:
            self.explore_dir = heading
            self.explore_until = turn + 8
        best = None
        best_info = ''
        for index, dest in enumerate(self.graph[origin]):
            if dest is None or dest in blocked:
                continue
            # Immediate legality used the old tail above; only now may it move.
            # If the full body is unseen, retaining it underestimates space.
            after = blocked | {dest}
            if dest not in self.pearls and self.tail is not None:
                after.remove(self.tail)
            direct, sprint, danger = self.threats(dest, after)
            distances = self.space(dest, after)
            exits = {q for q in self.graph[dest]
                     if q is not None and q not in after}
            safe_exits = exits - danger
            # Check room reachable through exits outside the predicted danger.
            safe_room = len(self.space(dest, after | danger))
            food_distance = min((distances[q] for q in self.pearls
                                 if q in distances), default=100)
            frontier_distance = min((distances[q] for q in targets
                                     if q in distances), default=100)
            exploring = food_distance == 100 or turn - self.last_meal >= 8
            # Nearby food still wins; prolonged scarcity promotes expansion
            # above distant food and heading, below every survival criterion.
            forage = (-food_distance if food_distance <= 2 or not exploring
                      else -3 - frontier_distance - 2 * self.visits.get(dest, 0))
            room = len(distances)
            required = min(ct.get_length() + 2, 20)
            # Lexicographic priorities: survival first, food once space is ample.
            score = (
                -direct, -sprint, bool(safe_exits), bool(exits),
                min(safe_room, required), min(room, required),
                food_distance <= 2, -food_distance if food_distance <= 2 else 0,
                self.sonar.movement_hint(self, dest, distances, DIRS[index])
                if exploring or self.sonar.request is not None else 0,
                forage, -frontier_distance if exploring else 0,
                min(len(safe_exits), 2),
                -self.visits.get(dest, 0),
                # Once survival, food and revisits tie, keep our course.
                # Extra room beyond the safety threshold should not cause jitter.
                DIRS[index] == heading, DIRS[index] == self.explore_dir,
                room, -((index - ct.get_id() % 4) % 4),
            )
            if best is None or score > best[0]:
                best = (score, DIRS[index])
                best_info = (f'v5 room={room} safe={safe_room} exits={len(exits)} '
                             f'risk={direct}/{sprint} food={food_distance} '
                             f'intel={len(self.sonar.reports)} breed_food={self.breeding.tier}')
        if best is not None:
            ct.set_indicator_string(best_info)
            ct.make_move(best[1])
            self.sonar.emit(self, ct, turn, origin, self.destination(origin, best[1]), DIRS)
        elif ct.can_split(2) and not retire:
            # Splitting is the only stationary legal action: preserve the
            # parent this turn and give a reversed child a chance to escape.
            child_size = self.breeding.split_size(self, ct, game, origin, blocked,
                                                   emergency=True)
            ct.set_indicator_string('v5 no verified move: emergency split '
                                    f'front={ct.get_length() - child_size} back={child_size}')
            ct.do_split(child_size)
            self.sonar.emit(self, ct, turn, origin, origin, DIRS)
        else:
            ct.set_indicator_string('v5 yield: crowded short snake, no rescue'
                                    if retire else
                                    'v5 no verified move: uncertain fallback')
            # Prefer a non-kelp edge with unverified destination to certain
            # body collision, when local information cannot certify a move.
            options = [d for d in DIRS
                       if self.tiles[origin].get_edge(d).get_edge_type()
                       != bc.EdgeType.KELP
                       and self.destination(origin, d) is None]
            ct.make_move(options[0] if options else DIRS[0])
            self.sonar.emit(self, ct, turn, origin, origin, DIRS)


def main():
    ct, game = bc.init()
    planner = Planner()
    while bc.update(ct, game):
        planner.choose(ct, game)
        bc.end_turn()


if __name__ == '__main__':
    main()

