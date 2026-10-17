"""Bridge a blocking token iterator to asyncio, and decode UTF-8 safely.

A token is not a character: byte-level BPE tokens may split a multi-byte UTF-8
sequence.  ``IncrementalUTF8Decoder`` keeps the partial sequence between feeds
so a stream never emits a replacement character at a token boundary.

``GenerationJob`` runs one blocking generator on a single-worker executor and
re-exposes it as an async iterator.  The executor is what serialises model
access; the job never lets a cancelled request keep mutating the model while a
newer request is running.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from concurrent.futures import Executor

logger = logging.getLogger(__name__)


class IncrementalUTF8Decoder:
    """Decode a stream of bytes into text without splitting UTF-8 sequences."""

    def __init__(self, errors: str = "replace") -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors)

    def feed(self, data: bytes) -> str:
        return self._decoder.decode(data)

    def finish(self) -> str:
        return self._decoder.decode(b"", final=True)


def _consume_future_exception(future) -> None:
    if future.cancelled():
        return
    with contextlib.suppress(BaseException):  # the consumer already saw it
        future.exception()


class StopBuffer:
    """Incrementally apply user ``stop`` strings to a text stream.

    The last ``max_stop_len - 1`` characters are held back so a stop sequence
    split across two token chunks is still detected; they are released by
    ``finish()`` when the stream ends without a match.
    """

    def __init__(self, stops: Sequence[str] | None = None) -> None:
        self.stops = tuple(dict.fromkeys(s for s in (stops or ()) if s))
        self._hold = max((len(s) for s in self.stops), default=0) - 1
        self._buf = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        if not self.stops:
            self._buf += text
            out, self._buf = self._buf, ""
            return out
        self._buf += text
        hit = self._find()
        if hit is not None:
            out = self._buf[:hit]
            self._buf = ""
            self.stopped = True
            return out
        if len(self._buf) > self._hold:
            cut = len(self._buf) - self._hold
            out, self._buf = self._buf[:cut], self._buf[cut:]
            return out
        return ""

    def finish(self) -> str:
        if self.stopped:
            return ""
        out, self._buf = self._buf, ""
        return out

    def _find(self) -> int | None:
        best: int | None = None
        for stop in self.stops:
            index = self._buf.find(stop)
            if index != -1 and (best is None or index < best):
                best = index
        return best


class GenerationJob[T]:
    """A running blocking generator, consumed as an async iterator.

    The iterator must check ``cancel`` itself (the engine does, once per
    token).  ``aclose()`` sets the flag and waits for the worker so callers can
    guarantee the model is no longer executing when a request ends.
    """

    def __init__(
        self,
        executor: Executor,
        iterator: Iterator[T],
        *,
        cancel: threading.Event | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self.cancel = cancel or threading.Event()
        self._future = self._loop.run_in_executor(executor, self._run, iterator)
        # Retrieve the exception even if the consumer goes away before aclose;
        # otherwise asyncio logs "Future exception was never retrieved".
        self._future.add_done_callback(_consume_future_exception)

    def _run(self, iterator: Iterator[T]) -> None:
        try:
            for item in iterator:
                self._put(("token", item))
        except BaseException as exc:  # surface engine errors to the consumer
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                logger.debug("generation failed", exc_info=True)
            self._put(("error", exc))
        finally:
            self._put(("done", None))

    def _put(self, item: tuple[str, object]) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:  # event loop is already closed
            logger.debug("event loop closed; dropping generation event")

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        kind, payload = await self._queue.get()
        if kind == "token":
            return payload  # type: ignore[return-value]
        if kind == "error":
            assert isinstance(payload, BaseException)
            raise payload
        raise StopAsyncIteration

    def cancel_now(self) -> None:
        """Ask the engine to stop at the next token boundary."""
        self.cancel.set()

    async def aclose(self) -> None:
        """Cancel and wait for the executor future to finish.

        Safe to call after normal completion; errors from the generator are
        swallowed here because they were already delivered through the queue.
        """
        self.cancel.set()
        future = asyncio.wrap_future(self._future, loop=self._loop)
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - already reported to the consumer
            logger.debug("generation future ended with an error", exc_info=True)
