"""How the tests choose ConcatFile's request shapes.

A ConcatFile made without ``shape=`` starts at a random shape. Here that shape comes from a
generator seeded by the test's name, so a failure repeats, and DBAUDIT_SEED reseeds every
test, to run the suite across other shapes.

``@pytest.mark.unshaped`` makes every lead 0, so each request asks exactly what ConcatFile
wants -- for a fake server that decides its answer by where a request starts or how long it
is, or by counting the fetches that cover an offset, and for a test that counts requested
bytes. ``--unshaped`` does that for every test, which sends the requests ConcatFile sent
before shapes: the tests that fail then are the ones that need shapes to pass.
"""

import gc
import os
import random
from types import SimpleNamespace

import pytest

from dbaudit.archive import reader
from tests import stores


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--unshaped",
        action="store_true",
        help="make every ConcatFile lead 0, as before request shapes",
    )


@pytest.fixture(autouse=True)
def _shapes(request: pytest.FixtureRequest):
    # A patch of its own: a test that calls monkeypatch.undo() must not undo this too.
    with pytest.MonkeyPatch.context() as patch:
        seed = f"{os.environ.get('DBAUDIT_SEED', '')}/{request.node.nodeid}"
        patch.setattr(
            reader, "secrets", SimpleNamespace(randbelow=random.Random(seed).randrange)
        )
        if request.config.getoption("--unshaped") or request.node.get_closest_marker(
            "unshaped"
        ):
            patch.setattr(reader, "LEAD_STEP", 0)
        yield


@pytest.fixture(autouse=True)
def _close_stores():
    """Close the stores the test handed to `tests.stores.close_after_test`, then
    collect garbage so that a connection this test leaked is reported against it."""
    yield
    stores.close_opened()
    gc.collect()
