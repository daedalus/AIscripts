#!/usr/bin/env python3
"""
Unified English generator — extended edition.

Modes:
  grammar   Probabilistic CFG with verb subcategorization & coordination
  energy    Simulated annealing with trigram energy model
  math      Polynomial word-index sequence (deterministic with --seed)
  hybrid    Grammar skeleton + annealing lexical substitution

Flags:
  --count N      Number of sentences (default 5)
  --seed N       Fix random seed for reproducible output
  --score        Print energy score alongside each sentence (energy/hybrid)
  --steps N      Annealing iterations (default 3000)
  --temp F       Initial annealing temperature (default 1.2)
"""

import random
import math
import argparse

# ──────────────────────────────────────────────
# Expanded vocabulary
# ──────────────────────────────────────────────

DETERMINERS_SING = ["the", "a", "this", "that", "every", "each", "no"]
DETERMINERS_PLUR = ["the", "these", "those", "some", "all", "no", "many"]
DETERMINERS      = DETERMINERS_SING  # flat list for energy/math vocab

ADJECTIVES = [
    "complex", "emergent", "mathematical", "unstable", "recursive",
    "hidden", "abstract", "discrete", "probabilistic", "sparse",
    "latent", "compressed", "adaptive", "nonlinear", "symmetric",
    "fragile", "singular", "irreducible", "coherent", "stochastic",
]

ADVERBS = [
    "quietly", "unexpectedly", "systematically", "partially",
    "recursively", "iteratively", "asymptotically", "gradually",
    "silently", "precisely", "blindly", "locally", "globally",
]

# Transitive verbs (require an object)
TRANS_VERBS = {
    "sing": ["observes", "builds", "analyzes", "generates", "encodes",
             "compresses", "extracts", "models", "approximates", "transforms",
             "minimizes", "maximizes", "samples", "decodes", "traverses"],
    "plur": ["observe", "build", "analyze", "generate", "encode",
             "compress", "extract", "model", "approximate", "transform",
             "minimize", "maximize", "sample", "decode", "traverse"],
}

# Intransitive verbs (no object needed)
INTRANS_VERBS = {
    "sing": ["converges", "diverges", "oscillates", "collapses",
             "emerges", "propagates", "stabilizes", "bifurcates"],
    "plur": ["converge", "diverge", "oscillate", "collapse",
             "emerge", "propagate", "stabilize", "bifurcate"],
}

NOUNS = [
    {"word": "model",        "number": "sing"},
    {"word": "system",       "number": "sing"},
    {"word": "language",     "number": "sing"},
    {"word": "structure",    "number": "sing"},
    {"word": "function",     "number": "sing"},
    {"word": "lattice",      "number": "sing"},
    {"word": "network",      "number": "sing"},
    {"word": "signal",       "number": "sing"},
    {"word": "automaton",    "number": "sing"},
    {"word": "distribution", "number": "sing"},
    {"word": "systems",      "number": "plur"},
    {"word": "patterns",     "number": "plur"},
    {"word": "algorithms",   "number": "plur"},
    {"word": "matrices",     "number": "plur"},
    {"word": "sequences",    "number": "plur"},
    {"word": "researchers",  "number": "plur"},
    {"word": "networks",     "number": "plur"},
    {"word": "layers",       "number": "plur"},
    {"word": "gradients",    "number": "plur"},
    {"word": "embeddings",   "number": "plur"},
]

OBJECTS = [
    "patterns", "the system", "a structure", "information",
    "hidden rules", "the gradient", "a representation",
    "the latent space", "sparse features", "the distribution",
    "a fixed point", "the residual", "local minima",
    "the eigenstructure", "recursive patterns",
]

# Shared flat vocabulary for energy / math generators
WORDS = (
    [d for d in DETERMINERS] +
    [a for a in ADJECTIVES[:10]] +
    [n["word"] for n in NOUNS] +
    list(TRANS_VERBS["sing"][:8]) +
    list(INTRANS_VERBS["sing"][:4]) +
    ADVERBS[:6]
)
WORDS = list(dict.fromkeys(WORDS))  # deduplicate, preserve order

# ──────────────────────────────────────────────
# 1. Grammar-based generator (PCFG)
# ──────────────────────────────────────────────

GRAMMAR = {
    # Coordination added: S can be two conjoined clauses
    "S":  [(["NP", "VP"],            0.70),
           (["NP", "VP", "and", "NP", "VP"], 0.30)],
    "NP": [(["Det", "N"],            0.50),
           (["Det", "Adj", "N"],     0.35),
           (["Det", "Adj", "Adj", "N"], 0.15)],
    # Verb type split: transitive vs intransitive
    "VP": [(["TV", "Obj"],           0.40),
           (["TV", "Adv", "Obj"],    0.25),
           (["TV", "Obj", "Adv"],    0.15),
           (["IV"],                  0.10),
           (["IV", "Adv"],           0.10)],
}


def weighted_choice(rules):
    total = sum(w for _, w in rules)
    r = random.uniform(0, total)
    upto = 0.0
    for item, weight in rules:
        upto += weight
        if r < upto:           # fixed: strict < avoids silent None
            return item
    return rules[-1][0]        # fallback for float rounding


def grammar_expand(symbol, ctx):
    """Recursively expand a grammar symbol into a list of words."""
    if symbol in GRAMMAR:
        rule = weighted_choice(GRAMMAR[symbol])

        # For NP rules, expand N first so ctx["number"] is set before Det
        if symbol == "NP":
            n_idx = next((i for i, s in enumerate(rule) if s == "N"), None)
            if n_idx is not None:
                noun_words = grammar_expand("N", ctx)
                result = []
                for i, sym in enumerate(rule):
                    if i == n_idx:
                        result.extend(noun_words)
                    else:
                        result.extend(grammar_expand(sym, ctx))
                return result

        result = []
        for sym in rule:
            result.extend(grammar_expand(sym, ctx))
        return result
        rule = weighted_choice(GRAMMAR[symbol])
        result = []
        for sym in rule:
            result.extend(grammar_expand(sym, ctx))
        return result

    if symbol == "Det":
        number = ctx.get("number", "sing")
        pool = DETERMINERS_PLUR if number == "plur" else DETERMINERS_SING
        return [random.choice(pool)]
    if symbol == "Adj":
        return [random.choice(ADJECTIVES)]
    if symbol == "N":
        noun = random.choice(NOUNS)
        ctx["number"] = noun["number"]   # note: coordination uses its own ctx
        return [noun["word"]]
    if symbol == "TV":
        return [random.choice(TRANS_VERBS[ctx.get("number", "sing")])]
    if symbol == "IV":
        return [random.choice(INTRANS_VERBS[ctx.get("number", "sing")])]
    if symbol == "Obj":
        return [random.choice(OBJECTS)]
    if symbol == "Adv":
        return [random.choice(ADVERBS)]
    # Literal tokens (e.g. "and")
    return [symbol]


def grammar_sentence():
    # Coordination: each clause gets its own context so number agreement
    # is tracked independently — fixes the shared-context mutation bug.
    rule = weighted_choice(GRAMMAR["S"])
    if "and" in rule:
        # Split at "and"
        idx = rule.index("and")
        left_syms, right_syms = rule[:idx], rule[idx+1:]
        ctx_l, ctx_r = {}, {}
        left  = [w for s in left_syms  for w in grammar_expand(s, ctx_l)]
        right = [w for s in right_syms for w in grammar_expand(s, ctx_r)]
        words = left + ["and"] + right
    else:
        ctx = {}
        words = [w for s in rule for w in grammar_expand(s, ctx)]

    sentence = " ".join(words).capitalize() + "."
    return sentence


# ──────────────────────────────────────────────
# 2. Energy-based generator (simulated annealing)
#    Now uses trigrams + richer bigram table
# ──────────────────────────────────────────────

BIGRAMS = {
    ("the",    "system"):      2.5,
    ("the",    "language"):    2.0,
    ("the",    "gradient"):    2.0,
    ("the",    "network"):     2.0,
    ("a",      "model"):       2.0,
    ("a",      "structure"):   2.0,
    ("a",      "function"):    2.0,
    ("this",   "lattice"):     2.0,
    ("each",   "layer"):       1.8,
    ("sparse", "features"):    2.5,
    ("latent", "space"):       2.5,
    ("hidden", "structure"):   2.5,
    ("complex","system"):      2.5,
    ("emergent","patterns"):   2.5,
    ("local",  "minima"):      2.0,
    ("system", "builds"):      2.0,
    ("system", "converges"):   2.0,
    ("language","generates"):  2.0,
    ("model",  "analyzes"):    2.0,
    ("model",  "approximates"):2.0,
    ("network","encodes"):     2.0,
    ("layers", "compress"):    2.0,
    ("builds", "structure"):   2.5,
    ("generates","patterns"):  2.5,
    ("analyzes","information"):2.5,
    ("encodes","a"):           1.5,
    ("extracts","hidden"):     2.5,
    ("gradually","stabilizes"):2.0,
    ("silently","diverges"):   2.0,
    ("recursively","builds"):  2.5,
    ("iteratively","refines"): 2.5,
}

TRIGRAMS = {
    ("the",     "latent",   "space"):     3.0,
    ("a",       "complex",  "system"):    3.0,
    ("the",     "hidden",   "structure"): 3.0,
    ("the",     "sparse",   "features"):  2.8,
    ("the",     "local",    "minima"):    2.8,
    ("this",    "emergent", "pattern"):   2.5,
    ("layers",  "silently", "converge"):  2.5,
    ("gradients","gradually","vanish"):   2.5,
    ("the",     "network",  "encodes"):   2.5,
    ("a",       "model",    "approximates"): 2.5,
}

ALL_VERBS_SET = (
    set(TRANS_VERBS["sing"]) |
    set(TRANS_VERBS["plur"]) |
    set(INTRANS_VERBS["sing"]) |
    set(INTRANS_VERBS["plur"])
)


def energy(seq):
    e = 0.0
    n = len(seq)

    # Trigram bonuses (checked first, most specific)
    trigram_positions = set()
    for i in range(n - 2):
        tri = (seq[i], seq[i+1], seq[i+2])
        if tri in TRIGRAMS:
            e -= TRIGRAMS[tri]
            trigram_positions.update([i, i+1, i+2])

    # Bigram bonuses (skip positions already covered by a trigram)
    for i in range(n - 1):
        if i in trigram_positions and i+1 in trigram_positions:
            continue
        pair = (seq[i], seq[i+1])
        if pair in BIGRAMS:
            e -= BIGRAMS[pair]
        else:
            e += 0.4   # reduced from 0.5 — less harsh on novel pairs

    # Length penalty (prefer 5–10 words)
    e += 0.08 * n
    if n < 4:
        e += 3.0
    if n > 12:
        e += 0.15 * (n - 12)

    # Must contain at least one verb
    if not any(w in ALL_VERBS_SET for w in seq):
        e += 3.0

    # Penalize immediate repetition
    for i in range(n - 1):
        if seq[i] == seq[i+1]:
            e += 1.5

    return e


def mutate(seq, words=WORDS):
    new_seq = seq[:]
    op = random.choices(
        ["replace", "insert", "delete", "swap"],
        weights=[0.40, 0.25, 0.20, 0.15]
    )[0]
    if op == "replace" and new_seq:
        i = random.randrange(len(new_seq))
        new_seq[i] = random.choice(words)
    elif op == "insert":
        i = random.randrange(len(new_seq) + 1)
        new_seq.insert(i, random.choice(words))
    elif op == "delete" and len(new_seq) > 4:   # raised floor from 1 to 4
        i = random.randrange(len(new_seq))
        del new_seq[i]
    elif op == "swap" and len(new_seq) > 1:
        i, j = random.sample(range(len(new_seq)), 2)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def energy_sentence(steps=3000, T0=1.2, words=WORDS, return_score=False):
    current = [random.choice(words) for _ in range(random.randint(5, 8))]
    current_e = energy(current)
    T = T0
    cooling = math.exp(math.log(0.01 / T0) / steps)  # reaches ~0.01 at end

    for _ in range(steps):
        candidate = mutate(current, words)
        candidate_e = energy(candidate)
        delta = candidate_e - current_e
        if delta < 0 or random.random() < math.exp(-delta / T):
            current, current_e = candidate, candidate_e
        T *= cooling

    sentence = " ".join(current).capitalize() + "."
    if return_score:
        return sentence, round(current_e, 3)
    return sentence


# ──────────────────────────────────────────────
# 3. Math-sequence generator
# ──────────────────────────────────────────────

def math_sequence(n_terms, vocab, offset=0):
    """
    Use polynomial (n²+n+1) mod N to index into vocab.
    N is computed from vocab at call time — no hardcoded default.
    offset shifts the starting n so successive sentences differ.
    """
    N = len(vocab)
    return [((n + offset) ** 2 + (n + offset) + 1) % N for n in range(n_terms)]


def math_sentence(n_terms=9, vocab=None, offset=0):
    if vocab is None:
        vocab = WORDS
    seq = math_sequence(n_terms, vocab, offset=offset)
    words = [vocab[i] for i in seq]

    # Insert object after transitive verbs
    trans_set = set(TRANS_VERBS["sing"]) | set(TRANS_VERBS["plur"])
    enriched = []
    for w in words:
        enriched.append(w)
        if w in trans_set and random.random() < 0.65:
            enriched.append(random.choice(OBJECTS))

    # Remove consecutive determiners (keep first) to reduce gibberish
    det_set = set(DETERMINERS_SING + DETERMINERS_PLUR)
    cleaned = []
    for w in enriched:
        if cleaned and w in det_set and cleaned[-1] in det_set:
            continue
        cleaned.append(w)

    return " ".join(cleaned).capitalize() + "."


# ──────────────────────────────────────────────
# 4. Hybrid generator (grammar skeleton + annealing)
# ──────────────────────────────────────────────

def hybrid_sentence(steps=1500, T0=0.8, return_score=False):
    """
    Generate a grammar skeleton, extract its slot vocabulary,
    then use annealing to substitute only the content words
    while keeping grammatical structure anchored.
    """
    ctx = {}
    # Get a flat word list from the grammar
    rule = weighted_choice(GRAMMAR["S"])
    # Avoid coordination for simplicity in hybrid mode
    while "and" in rule:
        rule = weighted_choice(GRAMMAR["S"])

    skeleton = [w for s in rule for w in grammar_expand(s, ctx)]

    # Build a per-position vocab: only substitute content words
    CONTENT_POS = set(ADJECTIVES) | set(ADVERBS) | set(OBJECTS) | \
                  {n["word"] for n in NOUNS} | \
                  set(TRANS_VERBS["sing"]) | set(TRANS_VERBS["plur"]) | \
                  set(INTRANS_VERBS["sing"]) | set(INTRANS_VERBS["plur"])

    # Indices of substitutable positions
    content_idx = [i for i, w in enumerate(skeleton) if w in CONTENT_POS]

    if not content_idx:
        sentence = " ".join(skeleton).capitalize() + "."
        return (sentence, 0.0) if return_score else sentence

    current = skeleton[:]
    current_e = energy(current)
    T = T0
    cooling = math.exp(math.log(0.01 / T0) / steps)

    for _ in range(steps):
        i = random.choice(content_idx)
        old_word = current[i]
        current[i] = random.choice(WORDS)
        new_e = energy(current)
        delta = new_e - current_e
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_e = new_e
        else:
            current[i] = old_word   # revert
        T *= cooling

    sentence = " ".join(current).capitalize() + "."
    if return_score:
        return sentence, round(current_e, 3)
    return sentence


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified English generator — extended edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mode",
        choices=["grammar", "energy", "math", "hybrid"],
        help="Which generator to use",
    )
    parser.add_argument("--count",  type=int,   default=5,   help="Number of sentences")
    parser.add_argument("--seed",   type=int,   default=None, help="Random seed")
    parser.add_argument("--score",  action="store_true",     help="Print energy scores")
    parser.add_argument("--steps",  type=int,   default=3000, help="Annealing steps")
    parser.add_argument("--temp",   type=float, default=1.2,  help="Initial temperature")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    for i in range(args.count):
        if args.mode == "grammar":
            print(grammar_sentence())

        elif args.mode == "energy":
            result = energy_sentence(steps=args.steps, T0=args.temp, return_score=args.score)
            if args.score:
                sent, score = result
                print(f"[{score:+.3f}] {sent}")
            else:
                print(result)

        elif args.mode == "math":
            print(math_sentence(offset=i * 7))

        elif args.mode == "hybrid":
            result = hybrid_sentence(steps=args.steps, T0=args.temp, return_score=args.score)
            if args.score:
                sent, score = result
                print(f"[{score:+.3f}] {sent}")
            else:
                print(result)


if __name__ == "__main__":
    main()