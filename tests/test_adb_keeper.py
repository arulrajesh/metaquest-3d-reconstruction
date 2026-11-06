"""The adb keeper restores wireless adb and reverse forwards after the Quest sleeps (fake adb, no device)."""

from mq3drecon.streaming.adb import AdbReverseKeeper

SERIAL = "192.168.0.86:5555"


class FakeAdb:
    def __init__(self):
        self.state = "device"  # "device", "offline", or None (gone)
        self.reverse: set[int] = set()
        self.reachable = True
        self.calls: list[list[str]] = []

    def __call__(self, args):
        self.calls.append(args)
        if args == ["devices"]:
            line = f"\n{SERIAL}\t{self.state}" if self.state else ""
            return 0, "List of devices attached" + line + "\n"
        if args[0] == "disconnect":
            self.state = None
            return 0, f"disconnected {SERIAL}\n"
        if args[0] == "connect":
            if not self.reachable:
                return 1, f"cannot connect to {SERIAL}: timed out\n"
            self.state = "device"
            return 0, f"connected to {SERIAL}\n"
        if args[-1] == "--list":
            return 0, "".join(f"host-1 tcp:{p} tcp:{p}\n" for p in sorted(self.reverse))
        if args[2] == "reverse":
            self.reverse.add(int(args[3].split(":")[1]))
            return 0, ""
        return 1, ""


def make(fake, now):
    return AdbReverseKeeper([8080], runner=fake, clock=lambda: now[0], on_change=lambda text: None, reconnect_interval_s=15.0)


def test_missing_forward_is_restored():
    fake, now = FakeAdb(), [0.0]
    keeper = make(fake, now)
    assert keeper.tick() == [f"headset {SERIAL}: adb reverse tcp:8080 restored"]
    assert fake.reverse == {8080}
    assert keeper.tick() == []  # already in place: nothing to do


def test_sleeping_headset_is_reconnected_and_forward_restored_on_wake():
    fake, now = FakeAdb(), [0.0]
    keeper = make(fake, now)
    keeper.tick()  # remembers the wireless serial
    fake.state, fake.reverse, fake.reachable = None, set(), False  # asleep: gone from adb, forward lost
    assert keeper.tick() == []  # connect attempted, fails
    now[0] = 5.0
    keeper.tick()
    assert sum(c[0] == "connect" for c in fake.calls) == 1  # rate-limited to once per 15 s
    fake.reachable = True  # woke up
    now[0] = 20.0
    assert keeper.tick() == [f"headset {SERIAL} reconnected", f"headset {SERIAL}: adb reverse tcp:8080 restored"]
    assert fake.state == "device" and fake.reverse == {8080}


def test_remembered_headset_is_reconnected_after_a_restart(tmp_path):
    remember = tmp_path / "headsets.json"
    fake, now = FakeAdb(), [0.0]
    AdbReverseKeeper([8080], runner=fake, clock=lambda: now[0], on_change=lambda t: None, remember_file=remember).tick()
    assert SERIAL in remember.read_text()
    # A new server process starts while the headset is asleep (gone from adb), then it wakes.
    fake.state, fake.reverse = None, set()
    keeper = AdbReverseKeeper([8080], runner=fake, clock=lambda: now[0], on_change=lambda t: None, remember_file=remember)
    assert keeper.tick() == [f"headset {SERIAL} reconnected", f"headset {SERIAL}: adb reverse tcp:8080 restored"]


def test_offline_entry_is_disconnected_before_reconnecting():
    fake, now = FakeAdb(), [0.0]
    keeper = make(fake, now)
    keeper.tick()
    fake.state, fake.reverse = "offline", set()
    changes = keeper.tick()
    assert ["disconnect", SERIAL] in fake.calls
    assert changes == [f"headset {SERIAL} reconnected", f"headset {SERIAL}: adb reverse tcp:8080 restored"]
