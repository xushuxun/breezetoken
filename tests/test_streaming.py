from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from breezetoken.streaming import GenerationJob, IncrementalUTF8Decoder, StopBuffer


def test_incremental_decoder_keeps_partial_codepoint() -> None:
    decoder = IncrementalUTF8Decoder()
    assert decoder.feed("你好".encode()[:1]) == ""
    assert decoder.feed("你好".encode()[1:3]) == "你"
    assert decoder.feed("好".encode()) == "好"
    assert decoder.finish() == ""


def test_stop_buffer_holds_across_chunks() -> None:
    buffer = StopBuffer(["STOP"])
    assert buffer.feed("hello ST") == "hello"
    assert buffer.feed("OP tail") == " "
    assert buffer.stopped
    assert buffer.finish() == ""


def test_stop_buffer_releases_tail_without_match() -> None:
    buffer = StopBuffer(["###"])
    assert buffer.feed("abcd") == "ab"  # the last len(stop)-1 chars are held
    assert not buffer.stopped
    assert buffer.finish() == "cd"


def test_stop_buffer_without_stops_is_transparent() -> None:
    buffer = StopBuffer([])
    assert buffer.feed("abc") == "abc"
    assert buffer.finish() == ""


async def test_generation_job_streams_and_closes() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = GenerationJob(executor, iter([1, 2, 3]))
        assert [item async for item in job] == [1, 2, 3]
        await job.aclose()


async def test_generation_job_cancel_serialises_next_job() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow(cancel: threading.Event):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            for i in range(100):
                if cancel.is_set():
                    return
                time.sleep(0.005)
                yield i
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        cancel = threading.Event()
        first = GenerationJob(executor, slow(cancel), cancel=cancel)
        assert await first.__anext__() == 0
        first.cancel_now()
        await first.aclose()

        second = GenerationJob(executor, iter([42]))
        assert [item async for item in second] == [42]
        await second.aclose()
    assert max_active == 1


async def test_generation_job_surfaces_errors() -> None:
    def broken():
        yield 1
        raise RuntimeError("boom")

    with ThreadPoolExecutor(max_workers=1) as executor:
        job = GenerationJob(executor, broken())
        assert await job.__anext__() == 1
        with pytest.raises(RuntimeError, match="boom"):
            await job.__anext__()
        await job.aclose()


def test_stop_buffer_earliest_match() -> None:
    buffer = StopBuffer(["bbb", "aa"])
    assert buffer.feed("xxaa") == "xx"
    assert buffer.stopped
