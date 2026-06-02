from replay_rag import MockReasoningProvider, TraceCapture, capture


def test_capture_returns_trace_and_response():
    cap = TraceCapture(provider=MockReasoningProvider())
    trace, resp = cap.solve("What is the sum of 12 and 7?")
    assert trace.problem.startswith("What")
    assert "19" in trace.answer
    assert resp.output_tokens > 0
    assert trace.metadata["scaffolded"] is False
    assert trace.trace_id  # 16-char hex
    assert len(trace.trace_id) == 16


def test_capture_records_history():
    cap = TraceCapture(provider=MockReasoningProvider())
    cap.solve("What is the sum of 1 and 2?")
    cap.solve("What is the sum of 3 and 4?")
    assert len(cap.captured) == 2
    assert cap.captured[0].trace_id != cap.captured[1].trace_id


def test_capture_invokes_callback():
    seen = []
    cap = TraceCapture(provider=MockReasoningProvider(), on_capture=seen.append)
    cap.solve("What is the sum of 5 and 6?")
    assert len(seen) == 1
    assert "11" in seen[0].answer


def test_capture_context_manager():
    with capture(MockReasoningProvider()) as cap:
        cap.solve("What is the difference of 10 and 4?")
    assert len(cap.captured) == 1
    assert "6" in cap.captured[0].answer


def test_capture_marks_scaffolded():
    cap = TraceCapture(provider=MockReasoningProvider())
    trace, _ = cap.solve("What is the sum of 1 and 2?", scaffold="prior trace text")
    assert trace.metadata["scaffolded"] is True
