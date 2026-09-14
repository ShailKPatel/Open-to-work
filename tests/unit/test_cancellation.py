from app.ingest.github.cancellation import clear, is_cancelled, request_cancel


def test_not_cancelled_by_default():
    assert is_cancelled("some-run-id") is False


def test_request_cancel_marks_it_cancelled():
    request_cancel("run-a")
    assert is_cancelled("run-a") is True
    clear("run-a")  # don't leak into other tests


def test_clear_resets_it():
    request_cancel("run-b")
    clear("run-b")
    assert is_cancelled("run-b") is False


def test_ids_are_independent():
    request_cancel("run-c")
    assert is_cancelled("run-d") is False
    clear("run-c")


def test_clear_on_never_requested_id_is_a_no_op():
    clear("never-requested")  # must not raise
    assert is_cancelled("never-requested") is False
