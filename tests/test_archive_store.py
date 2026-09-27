import pytest

from dbaudit.archive.parts import Part
from dbaudit.archive.store import ArchiveStore
from dbaudit.archive.tarwalk import Member, WalkResult


def parts(count=2, content="ab"):
    return [Part(idx=i, name=f"a.tar.a{chr(ord('a') + i)}", size=100, offset=100 * i,
                 path_display=f"/d/a.tar.a{chr(ord('a') + i)}", dbx_id=f"id:{i}",
                 rev="r", content_hash=content * 32) for i in range(count)]


def member(offset, name="f.bin", size=10):
    return Member(hdr_offset=offset, data_offset=offset + 512, size=size, type="0",
                  mode=0o644, mtime=1, uname="user", gname="proj00001",
                  dir="run", name=name, linkname="")


def new_store(tmp_path):
    store = ArchiveStore(tmp_path / "archives.db")
    store.init_schema()
    return store


def registered(tmp_path, count=2):
    """A store with one archive, its parts and its segments. Returns the first segment id."""
    store = new_store(tmp_path)
    archive_id = store.register("a.tar", "tar", "/d", "dropbox", parts(count))
    store.seed_segments(archive_id, parts(count))
    return store, archive_id, store.segments(archive_id)[0]["id"]


def seg(store, archive_id, segment_id):
    """The current row for one segment, found by id rather than assumed list position."""
    return next(s for s in store.segments(archive_id) if s["id"] == segment_id)


def test_registering_twice_is_the_same_archive(tmp_path):
    store = new_store(tmp_path)
    first = store.register("a.tar", "tar", "/d", "dropbox", parts())
    second = store.register("a.tar", "tar", "/d", "dropbox", parts())
    assert first == second


def test_re_registering_moved_parts_updates_their_paths(tmp_path):
    """The parts of v01p0_incomplete moved folders; the index must follow them."""
    store = new_store(tmp_path)
    archive_id = store.register("a.tar", "tar", "/d", "dropbox", parts())
    moved = [Part(**{**p.__dict__, "path_display": f"/new{p.path_display}"})
             for p in parts()]
    assert store.register("a.tar", "tar", "/new/d", "dropbox", moved) == archive_id
    assert store.parts_of(archive_id)[0].path_display.startswith("/new")


def test_different_content_is_a_different_archive(tmp_path):
    store = new_store(tmp_path)
    first = store.register("a.tar", "tar", "/d", "dropbox", parts())
    second = store.register("a.tar", "tar", "/d", "dropbox", parts(content="cd"))
    assert first != second


def test_a_batch_writes_members_and_the_cursor_together(tmp_path):
    store, archive_id, segment_id = registered(tmp_path)
    store.commit_batch(archive_id, segment_id, [member(0), member(1024)], 2048,
                       requests=3, bytes_fetched=4096)
    assert store.stats(archive_id)["n_members"] == 2
    assert store.segments(archive_id)[0]["cursor_offset"] == 2048
    assert store.get("a.tar")["requests"] == 3


def test_a_failed_batch_leaves_neither_rows_nor_cursor(tmp_path):
    store, archive_id, segment_id = registered(tmp_path)
    bad = Member(1024, 1536, "not-an-int", "0", 0, 0, "", "", "run", "b.bin", "")
    with pytest.raises(ValueError):
        store.commit_batch(archive_id, segment_id, [member(0), bad], 2048)
    assert store.stats(archive_id)["n_members"] == 0
    assert store.segments(archive_id)[0]["cursor_offset"] is None


def test_replaying_a_batch_does_not_duplicate(tmp_path):
    store, archive_id, segment_id = registered(tmp_path)
    store.commit_batch(archive_id, segment_id, [member(0), member(1024)], 2048)
    store.commit_batch(archive_id, segment_id, [member(0), member(1024)], 2048)
    assert store.stats(archive_id)["n_members"] == 2


def test_finishing_records_the_outcome(tmp_path):
    store, archive_id, segment_id = registered(tmp_path)
    store.commit_batch(archive_id, segment_id, [member(0)], 1024)
    store.finish(archive_id, WalkResult("complete", 1024, 1, ""))
    row = store.get("a.tar")
    assert row["state"] == "complete" and row["end_offset"] == 1024
    assert row["finished_at"] is not None


def test_find_members_returns_every_match(tmp_path):
    """A tar may hold the same path twice; cat must not guess between them."""
    store, archive_id, segment_id = registered(tmp_path)
    store.commit_batch(archive_id, segment_id, [member(0), member(4096)], 8192)
    assert len(store.find_members(archive_id, "run/f.bin")) == 2


def test_seeding_creates_one_segment_per_part(tmp_path):
    store, archive_id, _ = registered(tmp_path, count=3)
    segments = store.segments(archive_id)
    assert [s["scan_from"] for s in segments] == [0, 100, 200]
    assert [s["stop_at"] for s in segments] == [100, 200, None]
    # Segment 0 needs no scan and no confirmation: offset 0 is where a tar begins.
    assert segments[0]["first_header"] == 0 and segments[0]["joined"] == 1
    assert segments[1]["first_header"] is None and segments[1]["joined"] == 0


def test_claiming_a_segment_is_exclusive(tmp_path):
    store, archive_id, _ = registered(tmp_path)
    first = store.claim_segment(archive_id, "w1")
    second = store.claim_segment(archive_id, "w2")
    assert first["idx"] != second["idx"]
    # T6-18's second requirement: the caller sees the row's *post*-claim values
    # (state='walking', its own owner), not a stale pre-claim read.
    assert first["state"] == "walking" and first["owner"] == "w1"
    assert store.claim_segment(archive_id, "w3") is None


def test_repairing_a_segment_drops_only_its_own_members(tmp_path):
    """A segment whose start was wrong must leave nothing behind when it is re-walked."""
    store, archive_id, _ = registered(tmp_path)
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[0]["id"], [member(0)], 512)
    store.commit_batch(archive_id, segments[1]["id"], [member(4096), member(8192)], 8704)
    store.drop_members_between(archive_id, 4096, 8704)
    remaining = [r["hdr_offset"] for r in store.query_members(archive_id)]
    assert remaining == [0]


# ---- Tests added by task-6-rulings.md (T6-3, T6-5, T6-8, T6-9, T6-10, T6-11, T6-13, T6-16) ----


def test_replacing_a_part_marks_the_old_archive_stale(tmp_path):
    """Same name, same folder, different content: the old index of that path is stale.

    A same-named archive in a *different* folder is a different archive
    (FaultSZ03_Source.tar exists in many fault folders) and must be left alone.
    """
    store = new_store(tmp_path)
    first = store.register("a.tar", "tar", "/d", "dropbox", parts())
    elsewhere = store.register("a.tar", "tar", "/other", "dropbox", parts(content="ef"))
    second = store.register("a.tar", "tar", "/d", "dropbox", parts(content="cd"))

    assert store.get("a.tar")["id"] == second

    first_row = store.connect().execute(
        "SELECT state, detail FROM archives WHERE id=?", (first,)).fetchone()
    assert first_row["state"] == "stale"
    assert str(second) in first_row["detail"]

    elsewhere_row = store.connect().execute(
        "SELECT state FROM archives WHERE id=?", (elsewhere,)).fetchone()
    assert elsewhere_row["state"] != "stale"


def test_resetting_a_segment_drops_its_span_and_rearms_it(tmp_path):
    """A segment whose start was wrong is reset: dropped rows, new start, re-armed,
    its own member count zeroed, and the archive's confirmed frontier raised to
    match the corrected start (T6-8, T6-12)."""
    store, archive_id, _ = registered(tmp_path)
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[0]["id"], [member(0)], 512)
    store.commit_batch(archive_id, segments[1]["id"], [member(4096), member(8192)], 8704)

    store.reset_segment(segments[1]["id"], 4608)

    remaining = [r["hdr_offset"] for r in store.query_members(archive_id)]
    assert remaining == [0]  # segment 0's row is untouched
    row = seg(store, archive_id, segments[1]["id"])
    assert row["state"] == "pending" and row["joined"] == 1
    assert row["first_header"] == 4608 and row["cursor_offset"] is None
    assert row["members"] == 0
    assert store.get("a.tar")["cursor_offset"] == 4608


def test_retiring_a_segment_drops_its_span_and_marks_it_beyond(tmp_path):
    """A segment lying entirely past where the chain ended has nothing to contribute."""
    store, archive_id, _ = registered(tmp_path)
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[0]["id"], [member(0)], 512)
    store.commit_batch(archive_id, segments[1]["id"], [member(4096)], 4608)

    store.retire_segment(segments[1]["id"])

    remaining = [r["hdr_offset"] for r in store.query_members(archive_id)]
    assert remaining == [0]
    row = seg(store, archive_id, segments[1]["id"])
    assert row["state"] == "beyond" and row["members"] == 0 and row["joined"] == 0


def test_retiring_a_segment_clears_a_scan_failure_that_is_no_longer_one(tmp_path):
    """`beyond` is the normal case, not a fault: a tar's terminator is usually not in
    its last part, so a trailing segment legitimately has no header of its own. Its scan
    failed while its fate was still unknown; once the join proves it was never on the
    chain, leaving that error beside it tells an operator something is wrong when
    nothing is. The detail stays -- it still explains the row.
    """
    store, archive_id, _ = registered(tmp_path)
    segments = store.segments(archive_id)
    # A chain that crossed on one run and found nothing on the next.
    store.finish_segment(segments[1]["id"], WalkResult("crossed", 4608, 0, "crossed at 4608"))
    store.fail_segment(segments[1]["id"], "no header between here and the end")
    assert seg(store, archive_id, segments[1]["id"])["error"]

    store.retire_segment(segments[1]["id"])

    row = seg(store, archive_id, segments[1]["id"])
    assert row["state"] == "beyond"
    assert row["error"] is None
    assert row["detail"] == "crossed at 4608"


def test_releasing_a_segment_still_charges_what_it_read(tmp_path):
    """Symmetric with finish_segment and fail_segment, and for the same reason: reads
    that land after the last committed batch -- the batch a `--max-batches` stop refuses
    -- are charged to nothing otherwise. An interrupted run is the normal workflow here,
    so that undercount would be routine, and bytes_fetched / member_bytes is the
    headline efficiency number for the whole project.
    """
    store, archive_id, _ = registered(tmp_path)
    claimed = store.claim_segment(archive_id, "w1")
    store.commit_batch(archive_id, claimed["id"], [member(0)], 512,
                       requests=2, bytes_fetched=1024)

    store.release_segment(claimed["id"], requests=3, bytes_fetched=2048)

    row = store.get("a.tar")
    assert row["requests"] == 5
    assert row["bytes_fetched"] == 3072
    assert seg(store, archive_id, claimed["id"])["state"] == "pending"


def test_releasing_a_segment_returns_it_to_pending_with_cursor_intact(tmp_path):
    """A walk stopped early (Ctrl-C, --max-batches) must resume, not rescan."""
    store, archive_id, _ = registered(tmp_path)
    claimed = store.claim_segment(archive_id, "w1")
    store.commit_batch(archive_id, claimed["id"], [member(0)], 512)

    store.release_segment(claimed["id"])

    row = seg(store, archive_id, claimed["id"])
    assert row["state"] == "pending" and row["cursor_offset"] == 512
    again = store.claim_segment(archive_id, "w2")
    assert again["id"] == claimed["id"]


def test_start_walk_reclaims_walking_and_error_segments(tmp_path):
    """A killed run's segments must come back, or `index` can never finish."""
    store, archive_id, _ = registered(tmp_path)
    walking = store.claim_segment(archive_id, "w1")
    store.commit_batch(archive_id, walking["id"], [member(0)], 512)   # then the process died
    errored = store.claim_segment(archive_id, "w2")   # the only segment left pending
    store.fail_segment(errored["id"], "boom")

    store.start_walk(archive_id)

    assert seg(store, archive_id, walking["id"])["state"] == "pending"
    assert seg(store, archive_id, walking["id"])["cursor_offset"] == 512
    assert seg(store, archive_id, errored["id"])["state"] == "pending"
    assert store.get("a.tar")["state"] == "walking"
    assert store.claim_segment(archive_id, "w3") is not None
    assert store.claim_segment(archive_id, "w4") is not None
    assert store.claim_segment(archive_id, "w5") is None


def test_finish_segment_and_fail_segment_add_their_cost_to_the_archive(tmp_path):
    """Reads made after the last batch must still count toward bytes_fetched."""
    store, archive_id, segment_id = registered(tmp_path)
    other = store.segments(archive_id)[1]["id"]

    store.finish_segment(segment_id, WalkResult("complete", 100, 0, ""),
                         requests=2, bytes_fetched=1024)
    store.fail_segment(other, "boom", requests=3, bytes_fetched=2048)

    row = store.get("a.tar")
    assert row["requests"] == 5
    assert row["bytes_fetched"] == 3072


def test_seeding_twice_does_not_duplicate_or_disturb_a_walking_segment(tmp_path):
    """Re-registering an archive must never disturb a walk already in progress."""
    store, archive_id, _ = registered(tmp_path)
    claimed = store.claim_segment(archive_id, "w1")
    store.commit_batch(archive_id, claimed["id"], [member(0)], 512)

    store.seed_segments(archive_id, parts())

    segments = store.segments(archive_id)
    assert len(segments) == 2
    row = seg(store, archive_id, claimed["id"])
    assert row["state"] == "walking" and row["cursor_offset"] == 512


def test_stats_reports_covered_and_confirmed_for_a_half_walked_archive(tmp_path):
    """One finished segment, one mid-cursor, one untouched."""
    store, archive_id, _ = registered(tmp_path, count=3)
    segments = store.segments(archive_id)
    seg0, seg1 = segments[0]["id"], segments[1]["id"]

    store.finish_segment(seg0, WalkResult("complete", 100, 1, ""))   # whole span: [0, 100)

    store.set_segment_start(seg1, 120)
    store.mark_joined(seg1)                       # confirms the frontier up to 120
    store.commit_batch(archive_id, seg1, [], 150)  # mid-walk: cursor advanced to 150
    # segments[2] (scan_from=200, stop_at=None) is left untouched.

    stats = store.stats(archive_id)
    assert stats["confirmed"] == 120
    assert stats["covered"] == 150  # 100 (seg0, whole span) + 50 (seg1, 150 - 100)


# ---- Tests added by fix round 1 (task review Important findings 1 & 2, plus minor gaps) ----


def test_start_walk_returns_the_confirmed_frontier_and_logs_a_run_start_event(tmp_path):
    """Both start_walk's return value and its run_start event were untested: deleting
    the log_event call, or returning None instead of cursor_offset, left the suite
    green. `run_start_covered` is `covered` (segment progress), which is a different
    number from the `confirmed` frontier `start_walk` returns -- pin both."""
    store, archive_id, _ = registered(tmp_path)
    seg1 = store.segments(archive_id)[1]["id"]  # span [100, 200)
    store.set_segment_start(seg1, 140)
    store.mark_joined(seg1)  # raises archives.cursor_offset (the confirmed frontier) to 140

    returned = store.start_walk(archive_id)

    assert returned == 140
    stats = store.stats(archive_id)
    assert stats["confirmed"] == 140
    assert stats["run_started_at"] is not None
    assert stats["run_start_covered"] == 40  # seg1's own covered contribution: 140 - 100


def test_a_stale_archive_stays_stale_across_batches_and_finishing(tmp_path):
    """Task review Finding 1: `mark_stale`'s flag must survive the very next worker
    batch, a `start_walk` reclaim, and `finish` -- otherwise nothing ever reads it as
    a trust signal, and the spec's "kept but flagged" promise for a stale archive is
    hollow. Checked independently at each step so a regression in any one of the
    three call sites is pinned by its own assertion."""
    store, archive_id, segment_id = registered(tmp_path)
    store.mark_stale(archive_id, "fingerprint mismatch")

    store.commit_batch(archive_id, segment_id, [member(0)], 512)
    assert store.get("a.tar")["state"] == "stale"

    store.start_walk(archive_id)
    assert store.get("a.tar")["state"] == "stale"

    store.finish(archive_id, WalkResult("complete", 512, 1, ""))
    assert store.get("a.tar")["state"] == "stale"


def test_a_finished_stale_archive_keeps_its_stale_reason(tmp_path):
    """Ruling amendment to Finding 1: a flag the next write silently erases is not
    a flag, and the same argument applies to the reason behind it -- so `finish`
    must guard `detail` the same way it guards `state`. The walk's own outcome is
    not lost even so: it stays on `segments.detail` and in the `finish` event."""
    store, archive_id, segment_id = registered(tmp_path)
    store.mark_stale(archive_id, "fingerprint mismatch")

    store.finish(archive_id, WalkResult("corrupt", 512, 1, "bad header at 512"))

    row = store.get("a.tar")
    assert row["state"] == "stale"
    assert row["detail"] == "fingerprint mismatch"


def test_a_stale_archive_stays_stale_across_fail(tmp_path):
    """Out-of-scope finding from the re-review: `fail` had the same unguarded
    `state='error'` as `commit_batch`/`start_walk`/`finish` before their fix.
    Reachable path this closes: stale -> fail -> 'error' -> a later `index` run
    treats 'error' as retryable and re-walks it, mixing bytes from changed parts
    into an old index. The error message itself still writes unconditionally --
    knowing why a run failed is useful whatever the resulting state."""
    store, archive_id, _ = registered(tmp_path)
    store.mark_stale(archive_id, "fingerprint mismatch")

    store.fail(archive_id, "Dropbox request failed: 503")

    row = store.get("a.tar")
    assert row["state"] == "stale"
    assert row["error"] == "Dropbox request failed: 503"


def test_a_stale_archive_found_to_be_pax_keeps_its_stale_flag_and_reason(tmp_path):
    """Both can land in one run: `_run_index` re-fingerprints the parts after the walk
    and marks the archive stale before it records the pax refusal. The stale flag and
    its reason are guarded here as `finish` guards them. The refusal is not lost: it
    is logged as an event, and it is the segment's own error."""
    store, archive_id, _ = registered(tmp_path)
    store.mark_stale(archive_id, "fingerprint mismatch")

    store.mark_unsupported(archive_id, "a pax extended header applies to the member at 0")

    row = store.get("a.tar")
    assert row["state"] == "stale"
    assert row["detail"] == "fingerprint mismatch"
    logged = store.connect().execute(
        "SELECT detail FROM events WHERE archive_id=? AND kind='unsupported'",
        (archive_id,)).fetchall()
    assert [r[0] for r in logged] == ["a pax extended header applies to the member at 0"]


def test_reregistering_the_same_parts_clears_a_stale_flag(tmp_path):
    """The other half of Finding 1: register's ON CONFLICT branch never touched
    state, so an archive marked stale stayed stale forever even once the live parts
    hashed back to its identity again."""
    store = new_store(tmp_path)
    archive_id = store.register("a.tar", "tar", "/d", "dropbox", parts())
    store.mark_stale(archive_id, "fingerprint mismatch")
    assert store.get("a.tar")["state"] == "stale"

    again = store.register("a.tar", "tar", "/d", "dropbox", parts())

    assert again == archive_id
    row = store.get("a.tar")
    assert row["state"] == "registered" and row["detail"] is None


def test_reregistering_lifts_unsupported(tmp_path):
    """`unsupported` is terminal for `index`, but it must not be terminal for good: if it
    was ever reached wrongly, registering the parts again is the way back. A real pax
    archive loses nothing by it -- the next walk meets the pax header again."""
    store = new_store(tmp_path)
    archive_id = store.register("a.tar", "tar", "/d", "dropbox", parts())
    store.mark_unsupported(archive_id, "a pax extended header applies to the member at 0")

    store.register("a.tar", "tar", "/d", "dropbox", parts())

    row = store.get("a.tar")
    assert row["state"] == "registered" and row["detail"] is None


def test_replaying_a_batch_does_not_inflate_the_segment_member_count(tmp_path):
    """Task review Finding 2: `members=members+len(...)` double-counts a replay,
    even though the design guarantees replays ("a restart re-walks exactly that
    batch"). Recomputing over the segment's span is what actually stays exact."""
    store, archive_id, segment_id = registered(tmp_path)  # segment 0 spans [0, 100)
    store.commit_batch(archive_id, segment_id, [member(0), member(10)], 512)
    store.commit_batch(archive_id, segment_id, [member(0), member(10)], 512)  # replay

    assert seg(store, archive_id, segment_id)["members"] == 2


def _ended_in_segment_2(tmp_path):
    """Five segments: 0 crossed into 1, 1 crossed into 2, 2 ended `corrupt`, 3 retired
    `beyond` by the join, 4 with rows and `joined` left from an earlier run. The archive
    carries 2's verdict."""
    store, archive_id, _ = registered(tmp_path, count=5)      # spans 0, 100, 200..
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[0]["id"], [member(0)], 120)
    store.finish_segment(segments[0]["id"], WalkResult("crossed", 120, 1))
    store.set_segment_start(segments[1]["id"], 120)
    store.mark_joined(segments[1]["id"])
    store.commit_batch(archive_id, segments[1]["id"], [member(120), member(150)], 210)
    store.finish_segment(segments[1]["id"], WalkResult("crossed", 210, 2))
    store.set_segment_start(segments[2]["id"], 210)
    store.mark_joined(segments[2]["id"])
    store.commit_batch(archive_id, segments[2]["id"], [member(210)], 280)
    store.finish_segment(segments[2]["id"], WalkResult("corrupt", 280, 1, "junk at 280"))
    store.set_segment_start(segments[3]["id"], 330)
    store.commit_batch(archive_id, segments[3]["id"], [member(330)], 390)
    store.retire_segment(segments[3]["id"])
    store.set_segment_start(segments[4]["id"], 420)
    store.mark_joined(segments[4]["id"])
    store.commit_batch(archive_id, segments[4]["id"], [member(420)], 500)
    store.finish(archive_id, WalkResult("corrupt", 280, 4, "junk at 280"))
    return store, archive_id


def test_recheck_walks_again_from_the_segment_before_the_verdict(tmp_path):
    """The segment where the walk ended was handed its start by the crossing out of the
    one before; a derailed chain can cross at a wrong header that re-reading agrees with.
    So the chain is walked again from that earlier segment's own confirmed start; the
    segment where it ended keeps its start, but no longer joined -- the join checks it
    against the crossing, and scanning for it again would find the same header at a
    cost with no bound; and every segment after it waits, `beyond`, for the chain to
    cross that far."""
    store, archive_id = _ended_in_segment_2(tmp_path)

    again, ending = store.recheck(archive_id)

    assert (again["idx"], ending["idx"]) == (1, 2)
    assert [r["hdr_offset"] for r in store.query_members(archive_id)] == [0]
    rows = store.segments(archive_id)
    assert (rows[0]["state"], rows[0]["cursor_offset"]) == ("crossed", 120)   # untouched
    assert (rows[1]["state"], rows[1]["first_header"], rows[1]["joined"],
            rows[1]["cursor_offset"], rows[1]["exit_offset"]) == ("pending", 120, 1, None,
                                                                  None)
    assert (rows[2]["state"], rows[2]["first_header"], rows[2]["joined"],
            rows[2]["cursor_offset"], rows[2]["members"]) == ("pending", 210, 0, None, 0)
    for later in rows[3:]:
        assert (later["state"], later["first_header"], later["joined"],
                later["cursor_offset"], later["members"]) == ("beyond", None, 0, None, 0)
    archive = store.get("a.tar")
    assert (archive["state"], archive["detail"], archive["cursor_offset"]) == (
        "registered", None, 120)
    [event] = store.query("SELECT detail FROM events WHERE archive_id=? AND kind='recheck'",
                          (archive_id,))
    assert "corrupt" in event["detail"] and "junk at 280" in event["detail"]


def _walked_again_to_the_same_verdict(store, archive_id):
    """What walking segments 1 and 2 of `_ended_in_segment_2` again, from the same starts,
    leaves: the same crossing, the same rows, the same `corrupt` at 280."""
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[1]["id"], [member(120), member(150)], 210)
    store.finish_segment(segments[1]["id"], WalkResult("crossed", 210, 2))
    store.commit_batch(archive_id, segments[2]["id"], [member(210)], 280)
    store.finish_segment(segments[2]["id"], WalkResult("corrupt", 280, 1, "junk at 280"))
    store.mark_joined(segments[2]["id"])
    store.finish(archive_id, WalkResult("corrupt", 280, 4, "junk at 280"))


def test_a_recheck_that_reaches_the_same_verdict_steps_one_segment_further_back(tmp_path):
    """Review 7's B2. Walked again from segment 1's start, the chain ends where it did --
    which says nothing about segment 1's start itself: a stored tarball that fills a
    whole part hands a wrong one across two boundaries. So asking again about the same
    verdict walks one segment further back each time, down to segment 0, whose start
    nobody hands it."""
    store, archive_id = _ended_in_segment_2(tmp_path)
    assert store.recheck(archive_id)[0]["idx"] == 1
    _walked_again_to_the_same_verdict(store, archive_id)

    again, ending = store.recheck(archive_id)

    assert (again["idx"], ending["idx"]) == (0, 2)
    assert store.query_members(archive_id) == []
    rows = store.segments(archive_id)
    assert (rows[0]["state"], rows[0]["first_header"], rows[0]["joined"]) == (
        "pending", 0, 1)
    # Walked again, and joined to the chain again, from where they began.
    assert [(r["state"], r["first_header"], r["joined"], r["members"])
            for r in rows[1:3]] == [("pending", 120, 0, 0), ("pending", 210, 0, 0)]
    assert [r["state"] for r in rows[3:]] == ["beyond", "beyond"]
    assert store.get("a.tar")["cursor_offset"] == 0
    [_, event] = store.query("SELECT detail FROM events WHERE archive_id=? AND "
                             "kind='recheck' ORDER BY rowid", (archive_id,))
    assert "further back" in event["detail"]

    _walked_again_to_the_same_verdict(store, archive_id)
    store.commit_batch(archive_id, store.segments(archive_id)[0]["id"], [member(0)], 120)
    assert store.recheck(archive_id)[0]["idx"] == 0          # there is nothing further back


def test_a_recheck_that_reaches_another_verdict_starts_again_from_the_segment_before_it(
        tmp_path):
    """The step back is for a verdict that came back the same. One that changed -- here, the
    chain now ends in segment 2 somewhere else -- is asked again from segment 1, as a
    first recheck would."""
    store, archive_id = _ended_in_segment_2(tmp_path)
    store.recheck(archive_id)
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[1]["id"], [member(120), member(150)], 210)
    store.finish_segment(segments[1]["id"], WalkResult("crossed", 210, 2))
    store.commit_batch(archive_id, segments[2]["id"], [member(210), member(250)], 290)
    store.finish_segment(segments[2]["id"], WalkResult("corrupt", 290, 2, "junk at 290"))
    store.finish(archive_id, WalkResult("corrupt", 290, 4, "junk at 290"))

    again, ending = store.recheck(archive_id)

    assert (again["idx"], ending["idx"]) == (1, 2)


def test_a_recheck_of_segment_0_walks_it_again_from_offset_0(tmp_path):
    store, archive_id, _ = registered(tmp_path, count=2)
    segments = store.segments(archive_id)
    store.commit_batch(archive_id, segments[0]["id"], [member(0)], 60)
    store.finish_segment(segments[0]["id"], WalkResult("corrupt", 60, 1, "junk at 60"))
    store.finish(archive_id, WalkResult("corrupt", 60, 1, "junk at 60"))

    again, ending = store.recheck(archive_id)

    assert again["idx"] == ending["idx"] == 0
    rows = store.segments(archive_id)
    assert (rows[0]["state"], rows[0]["first_header"]) == ("pending", 0)
    assert (rows[1]["state"], rows[1]["first_header"]) == ("beyond", None)
    assert store.query_members(archive_id) == []


def test_crossing_into_a_retired_segment_puts_everything_retired_after_it_back(tmp_path):
    """After a recheck the chain may cross where the old verdict said it ended: the
    segment it crosses into is re-armed from its exit, and every one retired after it is
    scanned and walked again."""
    store, archive_id, _ = registered(tmp_path, count=4)
    segments = store.segments(archive_id)
    for segment in segments[1:]:
        store.retire_segment(segment["id"])

    store.rearm_beyond(archive_id, 2)

    rows = store.segments(archive_id)
    assert [r["state"] for r in rows] == ["pending", "beyond", "pending", "pending"]
    assert all(r["first_header"] is None and r["joined"] == 0 for r in rows[2:])


def test_recheck_leaves_a_stale_archive_to_register(tmp_path):
    """Only `register` lifts `stale`: the parts changed, and walking them again under the
    old registration would mix bytes from two different archives into one index."""
    store, archive_id = _ended_in_segment_2(tmp_path)
    store.mark_stale(archive_id, "a part changed")

    assert store.recheck(archive_id) is None
    assert store.get("a.tar")["state"] == "stale"
    assert [r["hdr_offset"] for r in store.query_members(archive_id)] == [0, 120, 150, 210,
                                                                          420]
