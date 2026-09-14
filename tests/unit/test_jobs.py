import threading
import time

from app.core import jobs


def test_start_runs_worker_and_stream_sees_final_state():
    def worker(job):
        job.set_state(stage="running", index=1, total=2)
        job.set_state(stage="done", index=2, total=2)

    started = jobs.start("k1", worker)
    assert started is True

    events = list(jobs.stream("k1", poll_interval=0.01))
    assert events[-1]["stage"] == "done"
    assert events[-1]["running"] is False


def test_start_is_a_no_op_while_already_running():
    release = threading.Event()
    entered = threading.Event()

    def worker(job):
        entered.set()
        job.set_state(stage="running")
        release.wait(timeout=2)
        job.set_state(stage="done")

    assert jobs.start("k2", worker) is True
    entered.wait(timeout=2)
    # Second start while the first is still blocked mid-run: no new thread,
    # existing job left untouched.
    assert jobs.start("k2", worker) is False
    assert jobs.snapshot("k2")["stage"] == "running"

    release.set()
    # let it finish
    for _ in range(200):
        if not jobs.snapshot("k2")["running"]:
            break
        time.sleep(0.01)
    assert jobs.snapshot("k2")["stage"] == "done"


def test_snapshot_of_unknown_key_is_none():
    assert jobs.snapshot("never-started") is None


def test_stream_of_unknown_key_yields_nothing():
    assert list(jobs.stream("also-never-started", poll_interval=0.01)) == []


def test_a_crashing_worker_lands_in_error_state():
    def worker(job):
        raise RuntimeError("boom")

    jobs.start("k3", worker)
    events = list(jobs.stream("k3", poll_interval=0.01))
    assert events[-1]["stage"] == "error"
    assert events[-1]["running"] is False
