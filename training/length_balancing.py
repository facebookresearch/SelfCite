# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
import random

def create_edited_reject_prediction(reject_prediction: str, chosen_prediction: str) -> str:
    """
    Ensure that each <statement> in both reject_prediction and chosen_prediction has an equal number of citations.
    If the reject citations are too few, add a sufficient number of "fake" citations to reject.
    """

    # === Step 1: Split both predictions into corresponding <statement> ===
    # Assume that both predictions are aligned in the number, order, and text content (excluding citations) of <statements>

    reject_statements    = re.findall(r"<statement>(.*?)</statement>", reject_prediction,  flags=re.DOTALL)
    chosen_statements    = re.findall(r"<statement>(.*?)</statement>", chosen_prediction,  flags=re.DOTALL)

    # If the number of <statements> does not match, return the original reject, or raise an error as needed
    if len(reject_statements) != len(chosen_statements):
        # You can either return directly or raise a ValueError("Statement count mismatch.")
        # print("Warning: Statement count mismatch. Return original reject prediction.")
        # print(f"Reject: {len(reject_statements)} vs. Chosen: {len(chosen_statements)}")
        return reject_prediction, chosen_prediction, {}, 0

    # === Step 2: Define helper functions to parse citations within each <statement> and rebuild <statement> ===

    def extract_text_and_cites(statement_text: str):
        """
        Separate (plain text) and (all citations) within a <statement>.
        Assume there is only one <cite>...</cite>, but it may contain multiple [start-end].
        If there are multiple <cite>..., you can extend the parser accordingly.
        Returns: (plain text, [ (start, end), (start, end), ... ])
        """
        # First, find the content within <cite>...</cite> (assuming only one cite block)
        cite_match = re.search(r"<cite>(.*?)</cite>", statement_text, flags=re.DOTALL)
        if not cite_match:
            # If there is no <cite>, consider the citation list as []
            return statement_text, []

        cite_content = cite_match.group(1)  # Content inside <cite>
        # Extract the plain text part of <statement> (remove <cite>...</cite>)
        # Here, simply replace <cite>...</cite> with an empty string
        text_without_cite = statement_text.replace(f"<cite>{cite_content}</cite>", "")

        # Parse citations in the form [num-num]
        pattern = r"\[(\d+)-(\d+)\]"
        pairs = re.findall(pattern, cite_content)
        cites = []
        for (start, end) in pairs:
            start_i = int(start)
            end_i   = int(end)
            if end_i - start_i > 10:
                print(f"Warning: too long citation: {start_i}-{end_i}")
                end_i = start_i + 5
            if len(cites) > 0 and cites[-1][1] == start_i - 1:
                cites[-1] = (cites[-1][0], end_i)
            else:
                cites.append((start_i, end_i))

        return text_without_cite.strip(), cites

    def rebuild_statement(text_without_cite: str, cites: list[tuple[int,int]]) -> str:
        """
        Reassemble the plain text and citation list into a <statement>...</statement> string.
        cites is a list of (start, end) tuples.
        """
        # First, convert cites to a string like [435-435][437-437]
        cites_str = "".join(f"[{s}-{e}]" for (s,e) in cites)
        # Wrap it within <cite>...</cite>
        if cites_str.strip():
            cite_block = f"<cite>{cites_str}</cite>"
        else:
            # If there are no citations, leave it empty
            cite_block = "<cite></cite>"

        # Reassemble the statement: plain text + cite_block
        return f"<statement>{text_without_cite}{cite_block}</statement>"

    # === Step 3: Define a function to generate "fake" citations ===
    def generate_fake_citation(chosen_cites: list[tuple[int,int]],
                               used_cites: set[tuple[int,int]],
                               how_many: int,
                               length: int) -> list[tuple[int,int]]:
        """
        Generate `how_many` unique fake citations that do not overlap with chosen_cites.
        Conditions:
          1. Citation span is 1 or 2 (e.g., [428-428] or [428-429])
          2. Must not be identical to or overlap with chosen_cites
          3. Must be located 5~10 before the minimum of chosen or 5~10 after the maximum of chosen
          4. Citations must not overlap with each other
        Returns a list of (start, end) tuples.
        """

        if how_many == 1:
            lengths = [length]
        elif how_many == 2:
            if length <= 2:
                # If length is 1 or 2, directly make [1, 1]
                lengths = [1, 1]
            else:
                random_split = random.randint(1, length-1)
                lengths = [random_split, length-random_split]
        else:
            if length <= how_many:
                # If length <= how_many, directly make [1, 1, ...]
                lengths = [1] * how_many
            else:
                # Randomly split length into how_many parts, e.g., length=5, how_many=3 → [1, 2, 2]
                # Directly select unique split points from 1~length-1
                lengths = []
                cut_points = random.sample(range(1, length), how_many-1)
                cut_points.sort()
                for i in range(how_many-1):
                    lengths.append(cut_points[i] - (cut_points[i-1] if i > 0 else 0))
                lengths.append(length - cut_points[-1])
                print(f"Splitting {length} into {how_many} parts: {lengths}")

        if not chosen_cites:
            # If there are no cites in chosen, define a range
            # Assume selecting randomly between [100-100] ~ [105-105]
            chosen_min = random.randint(100, 300)
            chosen_max = random.randint(chosen_min, 500)
        else:
            chosen_min = min(s for (s,e) in chosen_cites)
            chosen_max = max(e for (s,e) in chosen_cites)

        # Collect the generated fake citations
        result = []

        # Helper function to check overlap
        def is_overlap(a_start, a_end, b_start, b_end):
            # Check if [a_start, a_end] and [b_start, b_end] overlap
            return not (a_end < b_start or b_end < a_start)

        # Consider chosen_cites and used_cites (including existing reject cites) as forbidden
        all_forbidden = list(chosen_cites)  # Cannot overlap with chosen
        for fc in used_cites:
            if fc not in all_forbidden:
                all_forbidden.append(fc)

        # Start generating
        tries = 0
        max_tries = 1000  # Prevent infinite loop
        while len(result) < how_many and tries < max_tries:
            tries += 1

            # Decide to place before or after
            direction = random.choice(["before", "after"])
            # span = 1 or 2
            span = lengths[len(result)] #random.choice([1, 2])
            
            if direction == "before":
                # Place 5~10 before the minimum chosen
                start_lower_bound = max(chosen_min - span - 10, 0)
                start_upper_bound = max(chosen_min - span - 3, 0)
            else:
                # Place 5~10 after the maximum chosen
                start_lower_bound = chosen_max + 3
                start_upper_bound = chosen_max + 10

            if start_lower_bound < 0:
                start_lower_bound = 0
            
            if start_lower_bound > start_upper_bound:
                # Prevent range inversion if chosen_min is very small
                start_lower_bound, start_upper_bound = start_upper_bound, start_lower_bound
            
            # Randomly select a start
            start_c = random.randint(start_lower_bound, max(start_lower_bound, start_upper_bound))

            end_c = start_c + (span - 1)

            # Check for overlap with forbidden citations
            overlap_found = False
            for (fs, fe) in all_forbidden:
                if is_overlap(start_c, end_c, fs, fe):
                    overlap_found = True
                    break
            
            if not overlap_found:
                # Valid citation
                result.append((start_c, end_c))
                all_forbidden.append((start_c, end_c))

        return result

    def coverage_of(cites: list[tuple[int,int]]) -> int:
        """Calculate the total number of sentences covered by citations: sum(end - start + 1 for each)"""
        return sum((end - start + 1) for (start, end) in cites)

    def is_overlapping(c1, c2) -> bool:
        """Check if two (start, end) tuples overlap"""
        return not (c1[1] < c2[0] or c2[1] < c1[0])
    
    # === Step 4: Align and add citations to both statements lists ===
    edited_reject_statements = []
    edited_chosen_statements = []

    coverages = {
        'reject_before_edit': 0,
        'chosen_before_edit': 0,
        'reject_after_edit': 0,
        'chosen_after_edit': 0,
        'total_cites_reject_before_edit': 0,
        'total_cites_chosen_before_edit': 0,
        'total_cites_reject_after_edit': 0,
        'total_cites_chosen_after_edit': 0,
    }
    too_many_tries = False
    for s_reject, s_chosen in zip(reject_statements, chosen_statements):
        # 1) Parse the text and citations of both
        reject_text, reject_cites = extract_text_and_cites(s_reject)
        chosen_text, chosen_cites = extract_text_and_cites(s_chosen)

        coverage_of_reject = coverage_of(reject_cites)
        coverage_of_chosen = coverage_of(chosen_cites)
        coverage_diff = coverage_of_chosen - coverage_of_reject
        coverages['reject_before_edit'] += coverage_of_reject
        coverages['chosen_before_edit'] += coverage_of_chosen
        coverages['total_cites_reject_before_edit'] += len(reject_cites)
        coverages['total_cites_chosen_before_edit'] += len(chosen_cites)


        # 2) Check if the text (excluding citations) matches; handle discrepancies if necessary
        #   Here, simply assume they are aligned
        #   Also, assume that the entire <statement> has only one <cite>...</cite>
        #   (If there are multiple <cite>, you can extend the parser)
        reject_text = reject_text.strip()
        chosen_text = chosen_text.strip()
        if reject_text.strip() != chosen_text.strip():
            # If texts differ significantly, retain the original reject or raise an error
            print("Warning: Text mismatch. Keep original reject statement.")
            print(f"Reject: {reject_text} vs. Chosen: {chosen_text}")
            edited_reject_statements.append(s_reject)
            continue

        # 3) If the number of chosen_cites > reject_cites, add citations
        diff = len(chosen_cites) - len(reject_cites)
        if diff > 0:
            # Need to add `diff` "fake" citations
            used_cites = set(reject_cites)  # Record existing reject citations
            # Generate fake citations
            to_shrink = -100
            if coverage_diff > 0:
                span_budget = coverage_diff #+ random.randint(1, 3)
                to_shrink = diff - coverage_diff
            else:
                # Here, extra coverage causes reject's coverage to exceed chosen, so need to shrink existing reject_cites by (1 - coverage_diff)
                span_budget = diff #random.randint(1, 3)
                to_shrink = diff - coverage_diff

            if to_shrink > 0:
                print(f"Warning: reject: {reject_cites}, chosen: {chosen_cites}, diff: {coverage_diff}, to_shrink: {to_shrink}")
                to_shrink = 1 - coverage_diff
                for i in range(len(reject_cites)):
                    if coverage_of([reject_cites[i]]) > 1:
                        shrink = reject_cites[i][1] - reject_cites[i][0] if to_shrink > reject_cites[i][1] - reject_cites[i][0] else to_shrink
                        reject_cites[i] = (reject_cites[i][0], reject_cites[i][1] - shrink)
                        to_shrink -= shrink
                    if to_shrink == 0:
                        break
                    if to_shrink < 0:
                        print(f"Warning: over-shrink! to_shrink: {to_shrink}")
                        break
            fake_cites = generate_fake_citation(chosen_cites, used_cites, how_many=diff, length=span_budget)
            # Add fake_cites
            new_reject_cites = reject_cites + fake_cites
            if to_shrink != -100:
                print(f"Shrinked reject_cites: {new_reject_cites}")
        else:
            # No need to add
            no_matched = False
            # Start loop to generate adjustments
            tries = 0
            max_tries = 31  # Prevent infinite loop
            while not no_matched and tries < max_tries:
                tries += 1
                if tries > 100:
                    too_many_tries = True
                    print(f"Warning: too many tries: {tries}")
                if coverage_diff > 0:
                    cited_sents_dict = {}
                    for cite in chosen_cites:
                        for i in range(cite[0], cite[1]+1):
                            cited_sents_dict[i] = 1
                    # chosen coverage > reject coverage
                    # Modify existing reject_cites to match the coverage
                    # Can modify the first and last citations, splitting the coverage_diff randomly
                    new_reject_cites = []
                    if len(reject_cites) == 1:
                        first_cite = reject_cites[0]
                        if first_cite[0] - 1 not in cited_sents_dict and first_cite[1] + 1 not in cited_sents_dict:
                            first_diff = random.randint(0, coverage_diff)
                        elif first_cite[0] - 1 not in cited_sents_dict:
                            first_diff = coverage_diff
                        elif first_cite[1] + 1 not in cited_sents_dict:
                            first_diff = 0
                        else:
                            first_diff = random.randint(0, coverage_diff)
                        second_diff = coverage_diff - first_diff

                        new_first_cite = (first_cite[0] - first_diff, first_cite[1] + second_diff)
                        if new_first_cite not in chosen_cites:
                            no_matched = True
                        new_reject_cites.append(new_first_cite)
                    else:
                        first_cite = reject_cites[0]
                        last_cite = reject_cites[-1]
                        if first_cite[0] - 1 not in cited_sents_dict and last_cite[1] + 1 not in cited_sents_dict:
                            first_diff = random.randint(0, coverage_diff)
                        elif first_cite[0] - 1 not in cited_sents_dict:
                            first_diff = coverage_diff
                        elif last_cite[1] + 1 not in cited_sents_dict:
                            first_diff = 0
                        else:
                            first_diff = random.randint(0, coverage_diff)
                        last_diff = coverage_diff - first_diff
                        new_first_cite = (first_cite[0] - first_diff, first_cite[1])
                        new_last_cite = (last_cite[0], last_cite[1] + last_diff)
                        if new_first_cite not in chosen_cites and new_last_cite not in chosen_cites:
                            no_matched = True
                        new_reject_cites.append(new_first_cite)
                        new_reject_cites.extend(reject_cites[1:-1])
                        new_reject_cites.append(new_last_cite)
                elif coverage_diff < 0:
                    # chosen coverage < reject coverage
                    # Modify existing reject_cites to match the coverage
                    # Can modify the first and last citations, reducing the coverage_diff randomly
                    new_reject_cites = []
                    if len(reject_cites) == 1:
                        first_cite = reject_cites[0]
                        first_diff = random.randint(0, -coverage_diff)
                        second_diff = -coverage_diff - first_diff
                        new_first_cite = (first_cite[0] + first_diff, first_cite[1] - second_diff)
                        if new_first_cite not in chosen_cites:
                            no_matched = True
                        new_reject_cites.append(new_first_cite)
                    else:
                        first_cite = reject_cites[0]
                        last_cite = reject_cites[-1]
                        first_diff = random.randint(0, -coverage_diff)
                        last_diff = -coverage_diff - first_diff
                        new_first_cite = (first_cite[0] + first_diff, first_cite[1])
                        new_last_cite = (last_cite[0], last_cite[1] - last_diff)
                        if new_first_cite not in chosen_cites and new_last_cite not in chosen_cites:
                            no_matched = True
                        new_reject_cites.append(new_first_cite)
                        new_reject_cites.extend(reject_cites[1:-1])
                        new_reject_cites.append(new_last_cite)
                else:
                    new_reject_cites = reject_cites
                    no_matched = True

        # 4) Sort all new_reject_cites and chosen_cites
        new_reject_cites = sorted(new_reject_cites, key=lambda x: x[0])
        new_chosen_cites = sorted(chosen_cites, key=lambda x: x[0])

        # Calculate how much coverage each has
        coverage_of_reject = coverage_of(new_reject_cites)
        coverage_of_chosen = coverage_of(new_chosen_cites)
        coverages['reject_after_edit'] += coverage_of_reject
        coverages['chosen_after_edit'] += coverage_of_chosen
        coverages['total_cites_reject_after_edit'] += len(new_reject_cites)
        coverages['total_cites_chosen_after_edit'] += len(new_chosen_cites)

        # 5) Reassemble the <statement> string
        rebuilt = rebuild_statement(reject_text, new_reject_cites)
        edited_reject_statements.append(rebuilt)
        rebuilt_chosen = rebuild_statement(chosen_text, new_chosen_cites)
        edited_chosen_statements.append(rebuilt_chosen)


    # === Step 5: Reassemble all edited statements back into the entire reject_prediction ===
    edited_reject_prediction = "\n".join(edited_reject_statements)
    edited_chosen_prediction = "\n".join(edited_chosen_statements)
    return edited_reject_prediction, edited_chosen_prediction, coverages, too_many_tries#, coverage_of_chosen


# ------------------ Below is a simple test example ------------------
if __name__ == "__main__":

    reject_prediction_example = """<statement>Shuttle schedule info.<cite>[437-437]</cite></statement>
<statement>Some other statement with multiple cites.<cite>[435-435][437-437]</cite></statement>
<statement>Final statement.<cite></cite></statement>
"""

    chosen_prediction_example = """<statement>Shuttle schedule info.<cite>[435-440]</cite></statement>
<statement>Some other statement with multiple cites.<cite>[435-438][437-437][440-445]</cite></statement>
<statement>Final statement.<cite>[450-450]</cite></statement>
"""

    edited_reject, edited_chosen, coverages = create_edited_reject_prediction(reject_prediction_example, chosen_prediction_example)
    print("===== Edited Reject Prediction =====")
    print(edited_reject)
    print("===== Edited Chosen Prediction =====")
    print(edited_chosen)
    print("===== Coverages =====")
    print(coverages)
