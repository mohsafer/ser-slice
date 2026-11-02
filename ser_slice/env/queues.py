"""Per-slice FIFO queues with overflow, deadline expiry, and SLA accounting."""
from collections import deque


class SliceQueue:
    """FIFO queue of packets (bits-granular service within the head packet).

    - push: drops the packet (and reports the dropped bits) when the backlog
      would exceed b_max.
    - serve: drains up to rate*dt bits FIFO; sojourn times are recorded for
      fully served packets.
    - drain_deadlines: drops head packets whose deadline has passed without
      being fully served; each drop counts as a URLLC deadline violation.
    """

    def __init__(self, b_max_bits):
        self.b_max = float(b_max_bits)
        self.pkts = deque()

    @property
    def backlog(self):
        return sum(p.remaining for p in self.pkts)

    def push(self, pkt):
        if self.backlog + pkt.bits > self.b_max:
            return False
        self.pkts.append(pkt)
        return True

    def serve(self, rate_bps, dt, now):
        """Serve up to rate*dt bits; returns (served_bits, served_pkts, sojourn_sum)."""
        budget = rate_bps * dt
        served_bits = served_pkts = sojourn_sum = 0.0
        while self.pkts and budget > 1e-9:
            p = self.pkts[0]
            take = min(p.remaining, budget)
            p.remaining -= take
            budget -= take
            served_bits += take
            if p.remaining <= 1e-9:
                self.pkts.popleft()
                served_pkts += 1
                sojourn_sum += now - p.t_arr
        return served_bits, served_pkts, sojourn_sum

    def drain_deadlines(self, now):
        """Drop expired head packets; returns the number of violations."""
        expired = 0
        while self.pkts and self.pkts[0].deadline is not None and now > self.pkts[0].deadline:
            self.pkts.popleft()
            expired += 1
        return expired

    def hol_age(self, now):
        return now - self.pkts[0].t_arr if self.pkts else 0.0
