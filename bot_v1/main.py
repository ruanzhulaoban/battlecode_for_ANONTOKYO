"""Conservative local planning with bounded enemy sprint prediction."""
from collections import deque
import helper as bc

DIRS = bc.Direction.get_direction_list()


class Planner:
    def __init__(self):
        self.visits = {}
        self.portals = {}

    def snapshot(self, ct, game):
        self.width, self.height = game.get_map_size()
        self.me = ct.get_id()
        self.team = ct.get_team()
        self.tiles = {}
        self.occupied = {}
        self.heads = []
        self.pearls = set()
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
                if edge.get_edge_type() == bc.EdgeType.PORTAL:
                    key = self.edge_key(pos, d)
                    self.portals.setdefault(edge.get_portal_id(), set()).add(key)
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
            room = len(distances)
            required = min(ct.get_length() + 2, 20)
            # Lexicographic priorities: survival first, food once space is ample.
            score = (
                -direct, -sprint, bool(safe_exits), bool(exits),
                min(safe_room, required), min(room, required),
                -food_distance, min(len(safe_exits), 2),
                -self.visits.get(dest, 0), room,
                -((index - (ct.get_id() + game.get_round_num()) % 4) % 4),
            )
            if best is None or score > best[0]:
                best = (score, DIRS[index])
                best_info = (f'v1 room={room} safe={safe_room} exits={len(exits)} '
                             f'risk={direct}/{sprint} food={food_distance}')
        if best is not None:
            ct.set_indicator_string(best_info)
            ct.make_move(best[1])
        elif ct.can_split(2):
            # Splitting is the only stationary legal action: preserve the
            # parent this turn and give a reversed child a chance to escape.
            ct.set_indicator_string('v1 no verified move: emergency split')
            ct.do_split(2)
        else:
            ct.set_indicator_string('v1 no verified move: uncertain fallback')
            # Prefer a non-kelp edge with unverified destination to certain
            # body collision, when local information cannot certify a move.
            options = [d for d in DIRS
                       if self.tiles[origin].get_edge(d).get_edge_type()
                       != bc.EdgeType.KELP
                       and self.destination(origin, d) is None]
            ct.make_move(options[0] if options else DIRS[0])


def main():
    ct, game = bc.init()
    planner = Planner()
    while bc.update(ct, game):
        planner.choose(ct, game)
        bc.end_turn()


if __name__ == '__main__':
    main()
