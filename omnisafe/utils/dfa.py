"""
Deterministic Finite Automata Construction

"""

from itertools import chain, combinations
import re

class DFA:
    def __init__(self, mona_dfa_string):
        self.sink = set()

        states, aps, labels, q0, delta, acc, shape = self.parse_mona_dfa(mona_dfa_string)

        self.states = states
        self.aps = aps
        self.labels = labels
        self.q0 = q0
        self.delta = delta
        self.acc = acc
        self.shape = shape

        self.identify_rejecting_states()

    @staticmethod
    def formula_to_label(formula, APs):
        """
        Return all positive-label tuples consistent with the formula.
        
        Args:
            formula: string like "a & ~b & c"
            AP: iterable of all atomic propositions (e.g. ['a','b','c','d'])
        
        Returns:
            list of tuples, each a set of positive atoms that satisfy the formula
        """
        formula = formula.replace(" ", "")
        terms = formula.split("&") if formula else []
        
        required = {t for t in terms if not t.startswith("~") and t != ""}
        forbidden = {t[1:] for t in terms if t.startswith("~")}
        
        # Candidate atoms are AP minus required and forbidden
        free = [a for a in APs if a not in required and a not in forbidden]
        
        labels = []
        # for each subset of the free atoms, add to the required ones
        for r in range(len(free)+1):
            for subset in combinations(free, r):
                label = required.union(subset)
                labels.append(tuple(sorted(label)))
        
        # Deduplicate while preserving order
        labels = list(dict.fromkeys(labels))
        return labels

    def parse_mona_dfa(self, mona_str):
        # Find number of states
        state_numbers = re.findall(r'(\d+) -> (\d+)', mona_str)
        unique_states = set(int(num) for pair in state_numbers for num in pair)
        states = list(sorted(unique_states))
        n_qs = len(states)

        # Find all strings after "label="
        edge_labels = re.findall(r'label="([^"]+)"', mona_str)

        # Find all atomic propositions (alphabet)
        aps = set()
        for label in edge_labels:
            aps.update(re.findall(r'\b[a-zA-Z_]\w*\b', label))
        aps = list(sorted(aps))

        # Build labels (power set of alphabet)
        labels = list(chain.from_iterable(combinations(aps, k) for k in range(len(aps)+1)))
        
        # The transition function
        delta = [{label:[] for label in labels} for i in range(n_qs)]

        lines = mona_str.splitlines()
        for line in lines:
            if 'shape = doublecircle' in line:
                acc = set(map(int, re.findall(r'\d+', line)))
            elif 'init ->' in line:
                q0 = int(re.search(r'init -> (\d+)', line).group(1))
            elif '->' in line:
                match = re.match(r'(\d+) -> (\d+) \[label="(.+?)"\]', line)
                if match:
                    """
                    TO-DO: deal with 'or' formulas
                    """
                    from_state, to_state, labels_formula = int(match.group(1)), int(match.group(2)), self.formula_to_label(match.group(3), aps)
                    from_idx = states.index(from_state)
                    to_idx = states.index(to_state)

                    for label in labels_formula:
                        delta[from_idx][label] = to_idx
        
        n_accs = len(acc)
        shape = n_accs, n_qs

        output = (states, aps, labels, q0, delta, acc, shape)
        return output

    def identify_rejecting_states(self):
        # Rejecting state is a "sink state", i.e. it only tansforms to itself and is not accepting
        for state in self.states:
            is_sink = True
            state_idx = self.states.index(state)
            for label in self.labels:
                next_state_idx = self.delta[state_idx][label]
                if next_state_idx != state_idx:
                    is_sink = False
                    break

            if is_sink and state not in self.acc:
                self.sink.add(state)

    def step(self, current_state, context):
        """
        Returns the next DFA state based on the current state and context (a dictionary with values for `a` and `b`).
        """
        for label in self.delta[current_state]:
            if self.evaluate_condition(label, context):
                return [self.delta[current_state][label]]
        return [current_state]  # If no condition matches, remain in current state

    @staticmethod
    def evaluate_condition(label, context):
        """
        Evaluates the logical condition in `label` using the `context` dictionary, where context = {'a': bool, 'b': bool}.
        """
        # Translate logical operators
        # condition = label.replace("~", "not ").replace("&", "and").replace("|", "or")
        
        try:
            # Safely evaluate the expression using the context
            if not label:  # empty tuple
                return False
            # If the atom is in the label, check its value in the context
            # Only True if all atom in the label is true
            return all(context.get(atom, False) for atom in label)
        except Exception as e:
            print(f"Error evaluating condition {label}: {e}")
            return False
        
    def get_dfa_reward(self, dfa_state):
        return 1 if dfa_state in self.acc else 0
