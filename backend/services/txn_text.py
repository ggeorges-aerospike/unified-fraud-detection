"""Memos and payee names for generated transfers (pure functions, seeded by the caller's RNG).

Real transfers carry text a customer typed: a memo and the payee name they entered. That text is
where judgments live that arithmetic can't make — a $40k transfer is fine if it is a home
closing; a $2k transfer is not fine if the memo is "gift cards for the IRS"; a payment to someone
else's account typed as the customer's own name is a classic mule cash-out.

Every transfer gets a memo and a payee name, ordinary ones included, so the mere presence of text
(or a payee name that differs from the account holder's) is never the giveaway.
"""

from __future__ import annotations

import random
import re
from typing import List

# --------------------------------------------------------------------------- ordinary
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_ORDINARY = [
    "Rent – {month}", "Rent share {month}", "Utilities split", "Electric bill share", "Dinner split",
    "Groceries", "Birthday gift", "Concert tickets", "Paying you back", "Gas money", "Babysitting",
    "Dog walking – {month}", "Phone bill", "Internet bill share", "Lunch", "Book club dues",
    "Soccer league fees", "Moving help – thanks!", "Car insurance share", "Wedding gift",
    "Invoice #{n} – lawn care", "Piano lessons – {month}", "Cleaning service", "Tutoring – {month}",
    "", "",
]
_TITLES = {"mr", "mrs", "ms", "miss", "dr", "phd", "md", "dds", "dvm", "jr", "sr", "ii", "iii", "iv"}
_NICKNAMES = {
    "michael": "Mike", "william": "Will", "robert": "Rob", "james": "Jim", "jennifer": "Jen",
    "elizabeth": "Liz", "christopher": "Chris", "matthew": "Matt", "joshua": "Josh", "daniel": "Dan",
    "jessica": "Jess", "katherine": "Kate", "kaitlyn": "Katie", "anthony": "Tony", "nicholas": "Nick",
    "jonathan": "Jon", "stephanie": "Steph", "rebecca": "Becky", "benjamin": "Ben", "samuel": "Sam",
    "alexander": "Alex", "patricia": "Pat", "richard": "Rick", "thomas": "Tom", "joseph": "Joe",
    "margaret": "Maggie", "andrew": "Andy", "timothy": "Tim", "zachary": "Zach", "victoria": "Vicky",
}


def _name_parts(name: str) -> List[str]:
    return [p for p in re.split(r"\s+", (name or "").strip()) if p and p.lower().strip(".,") not in _TITLES]


def ordinary_memo(rng: random.Random) -> str:
    return rng.choice(_ORDINARY).format(month=rng.choice(_MONTHS), n=rng.randint(1000, 9999))


def payee_as_typed(real_name: str, rng: random.Random) -> str:
    """How a customer might type a payee they know: usually the full name, sometimes informally."""
    parts = _name_parts(real_name)
    if len(parts) < 2:
        return real_name
    first, last = parts[0], parts[-1]
    r = rng.random()
    if r < 0.70:
        return f"{first} {last}"
    if r < 0.82:
        return f"{_NICKNAMES.get(first.lower(), first)} {last}"
    if r < 0.92:
        return first
    return f"{first[0]}. {last}"


# --------------------------------------------------------------------------- legit large (benign)
_STREETS = ["Elm St", "Maple Ave", "Oak Ridge Dr", "Lakeview Ct", "Cedar Ln", "Birch Rd", "Harbor Way"]
_CARS = ["Honda CR-V", "Toyota RAV4", "Subaru Outback", "Ford F-150", "Mazda CX-5", "Kia Telluride"]
_SCHOOLS = ["UT Austin", "Ohio State", "Penn State", "Arizona State", "UC Davis", "Purdue"]


def legit_large_memo(rng: random.Random) -> str:
    return rng.choice([
        "Closing funds – {n} {street}, escrow #{e}",
        "Down payment – {year} {car}, deal #{e}",
        "Tuition – {school} spring semester",
        "Q4 estimated federal tax payment",
        "Contractor final payment – kitchen remodel, invoice #{e}",
        "Wedding venue final balance – {month} {d} reception",
        "Security deposit + first month – {n} {street} Apt {a}",
        "Roof replacement – invoice #{e}",
    ]).format(n=rng.randint(10, 9999), street=rng.choice(_STREETS), e=rng.randint(10000, 99999),
              year=rng.randint(2022, 2026), car=rng.choice(_CARS), school=rng.choice(_SCHOOLS),
              month=rng.choice(_MONTHS), d=rng.randint(1, 28), a=rng.randint(1, 40))


# --------------------------------------------------------------------------- scams (fraud)
_SCAM_STORIES = [
    "Gift cards for IRS settlement – case #{n}",
    "Crypto wallet recovery fee",
    "Bail for grandson – urgent, don't tell mom",
    "Investment – guaranteed 30% monthly return",
    "Customs release fee for your package",
    "Refund reversal – tech support",
    "Plane ticket + visa fees for {first}, see you soon",
    "Social Security account protection transfer",
    "Lottery winnings processing fee",
]


def scam_story(rng: random.Random) -> str:
    """One victim's story; the victim reuses it across their transfers with small variations."""
    return rng.choice(_SCAM_STORIES)


def scam_memo(story: str, rng: random.Random) -> str:
    memo = story.format(n=rng.randint(100000, 999999), first=rng.choice(["Daniel", "Marco", "James", "Liam"]))
    return memo if rng.random() < 0.7 else f"{memo} (part {rng.randint(2, 3)})"


def vague_memo(rng: random.Random) -> str:
    return rng.choice(["", "payment", "transfer", "inv", "ok", "for you", "as discussed", "funds"])


# --------------------------------------------------------------------------- alias (fraud)
def alias_of(name: str, rng: random.Random) -> str:
    """A believable variant of the customer's own name, typed as the payee on a mule cash-out."""
    parts = _name_parts(name)
    if len(parts) < 2:
        return name
    first, last = parts[0], parts[-1]
    variants = [
        f"{first} {last[:-2] + last[-1] + last[-2] if len(last) > 3 else last + 'e'}",   # swapped letters
        f"{first[:-1] if len(first) > 3 else first} {last}",                            # dropped letter
        f"{_NICKNAMES.get(first.lower(), first[0] + '.')} {last}",                      # nickname / initial
        f"{last} {first}",                                                               # swapped order
        f"{first} {last} Savings",                                                       # "my other account"
    ]
    return rng.choice(variants)


def alias_memo(rng: random.Random) -> str:
    return rng.choice(["Moving savings", "To my other account", "Transfer to self", "Emergency fund",
                       "Savings", "", "My new bank"])
