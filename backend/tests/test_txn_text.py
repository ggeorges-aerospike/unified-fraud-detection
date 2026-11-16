"""Generated memo / payee text: varied, and only the planted patterns carry the tell-tale text."""

import random

from services import txn_text


def test_payee_as_typed_is_usually_the_real_name_and_never_the_senders():
    rng = random.Random(1)
    typed = [txn_text.payee_as_typed("Michael Waters", rng) for _ in range(300)]
    assert typed.count("Michael Waters") > 150
    assert {"Mike Waters", "Michael", "M. Waters"} <= set(typed)


def test_alias_is_a_variant_not_the_exact_name():
    rng = random.Random(2)
    for _ in range(50):
        a = txn_text.alias_of("Kaitlyn Castaneda PhD", rng)
        assert a != "Kaitlyn Castaneda" and ("Castaneda" in a or "Castanead" in a or "Kaitlyn" in a)


def test_ordinary_memos_never_read_like_scams():
    rng = random.Random(3)
    words = ("gift card", "crypto", "bail", "guaranteed", "irs", "lottery", "customs")
    for _ in range(500):
        m = txn_text.ordinary_memo(rng).lower()
        assert not any(w in m for w in words), m


def test_legit_large_memos_are_specific_and_vague_memos_are_not():
    rng = random.Random(4)
    assert all(len(txn_text.legit_large_memo(rng)) > 15 for _ in range(50))
    assert all(len(txn_text.vague_memo(rng)) <= 12 for _ in range(50))
