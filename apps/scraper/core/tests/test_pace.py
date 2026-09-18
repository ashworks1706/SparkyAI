from scraper.ingest.pace import HostPacer


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.now += secs


def test_a_second_fetch_to_the_same_host_waits_out_the_gap() -> None:
    clock = Clock()
    pacer = HostPacer(5.0, clock=clock.time, sleep=clock.sleep)

    pacer.wait("https://registrar.asu.edu/grades")
    clock.now += 2.0
    pacer.wait("https://registrar.asu.edu/transcripts")

    assert clock.slept == [3.0]


def test_hosts_are_paced_apart_and_a_zero_gap_never_waits() -> None:
    clock = Clock()
    pacer = HostPacer(5.0, clock=clock.time, sleep=clock.sleep)
    pacer.wait("https://registrar.asu.edu/grades")
    pacer.wait("https://tuition.asu.edu/cost")
    assert clock.slept == []

    off = HostPacer(0.0, clock=clock.time, sleep=clock.sleep)
    off.wait("https://a.asu.edu/")
    off.wait("https://a.asu.edu/")
    assert clock.slept == []
