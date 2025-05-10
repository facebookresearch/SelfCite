# Copyright (c) Meta Platforms, Inc. and affiliates.

def trim_sents_by_key_distance(sents, sent_lens, key_sents_ids, L):
    """
    Given:
        sents           : List[str], list of all sentences
        sent_lens       : List[int], lengths of each sentence
        key_sents_ids   : Set[int], set of indices for key sentences
        L               : int, length limit (e.g., 1024)

    Returns:
        List[str], list of trimmed sentences (maintaining original order)
    """

    n = len(sents)
    assert len(sent_lens) == n, "Length of sents and sent_lens do not match!"

    # 1. If the length limit is already met, return immediately
    total_len = sum(sent_lens)
    if total_len <= L:
        return sents, set()

    # 2. Calculate the index distance of each sentence to the nearest key sentence
    dist = [float('inf')] * n
    for i in range(n):
        if i in key_sents_ids:
            dist[i] = 0

    # 2a. Scan from left to right
    last_key = None
    for i in range(n):
        if dist[i] == 0:
            last_key = i
        elif last_key is not None:
            dist[i] = min(dist[i], i - last_key)

    # 2b. Scan from right to left
    last_key = None
    for i in reversed(range(n)):
        if dist[i] == 0:
            last_key = i
        elif last_key is not None:
            dist[i] = min(dist[i], last_key - i)

    # 3. Collect indices of all non-key sentences and sort them by distance in descending order
    non_key_indices = [i for i in range(n) if i not in key_sents_ids]
    non_key_indices.sort(key=lambda i: dist[i], reverse=True)

    # 4. Remove non-key sentences with the largest distance one by one until total length is <= L
    removed = set()
    i = 0
    while total_len > L and i < len(non_key_indices):
        idx = non_key_indices[i]
        total_len -= sent_lens[idx]
        removed.add(idx)
        i += 1

    # 5. Retain sentences that were not removed in their original order (key sentences are always retained)
    trimmed_sents = [sents[i] for i in range(n) if i not in removed]

    return trimmed_sents, removed


# ------------------Below is a simple test example------------------
if __name__ == "__main__":
    # Example: Each sentence length is simply represented by numbers (in a real scenario, it could be tokens/word count, etc.)
    sents_example = [
        "Sentence 0", "Sentence 1 (Key)", "Sentence 2", "Sentence 3", 
        "Sentence 4 (Key)", "Sentence 5", "Sentence 6"
    ]
    sent_lens_example = [100, 300, 100, 150, 200, 100, 250]
    key_sents_ids_example = {1, 4}  # Sentence 1 and Sentence 4 are key
    L_example = 800

    result = trim_sents_by_key_distance(
        sents_example,
        sent_lens_example,
        key_sents_ids_example,
        L_example
    )

    print("Trimmed Sentences:", result)
    print("Total Length =", sum(sent_lens_example[i] for i, s in enumerate(sents_example) if s in result))
