import string
import pathlib

import nltk
import numpy
import datasets
import nltk.corpus

from src.evaluation.dataset import DatasetState

class LexicalDiversityMeasure:
    """ Computes Lexical Diversity for a list of texts, as the TTR value. """

    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        self.stoplist = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))

    def tokenize(self, texts: list[str], filter_stopwords=True):
        texts = [ nltk.word_tokenize(text.lower()) for text in texts ]
        if filter_stopwords:
            texts = [ list(filter(lambda x: x not in self.stoplist, text)) for text in texts ]
        return texts

    def evaluate(self, texts):
        texts = self.tokenize(texts)

        types  = len(set.union(set(), *texts))
        tokens = sum(len(text) for text in texts)

        return types / tokens

def describe(dataset_state: DatasetState, lex_div: LexicalDiversityMeasure):
    print("[>] Properties of", dataset_state.name, f"({dataset_state.dtype})")
    query_data   = { concept: data['queries'] for concept, data in dataset_state.items() }
    num_queries  = sum(len(qdata) for qdata in query_data.values())

    max_depth = max(len(path) for path in dataset_state.values('node_paths'))
    diversity = numpy.array([ lex_div.evaluate(prompts) for prompts in query_data.values() ])

    num_leaves, num_ancestors, num_children = 0, 0, 0
    for key, node in dataset_state.items('node_data'):
        if not node['children']: num_leaves    += 1
        else: num_children  += len(node['children'])
        num_ancestors += (len(dataset_state.node_paths.deepget(key)) - 1)

    print("    Number of Concepts           :", len(query_data))
    print("    Number of Questions          :", num_queries)
    print("    Maximum Depth                :", max_depth)
    print("    Number of Leaves             :", num_leaves)
    print("    Average Ancestors            :", f"{num_ancestors / len(query_data):.1f}")
    print("    Average Children (non-leaves):", f"{num_children / (len(query_data) - num_leaves):.1f}")
    print("    Lexical Diversity (TTR)      :", f"{diversity.mean():.2%}")
    print()

def main():
    # change to wherever your copy of SELECT is
    root    = pathlib.Path("data")

    lex_div = LexicalDiversityMeasure()

    # SELECT: atom concepts

    select_atom = DatasetState("SELECT", root=root, compose=False)
    describe(select_atom, lex_div)

    # SELECT: compositions of concepts

    select_compose = DatasetState("SELECT", root=root, compose=True)
    describe(select_compose, lex_div)

    # Reference: OR-Bench (or-bench-hard-1k)

    or_bench = datasets.load_dataset("bench-llm/or-bench", 'or-bench-hard-1k')['train']
    or_bench = {
        category: or_bench.filter(lambda x: x['category'] == category)['prompt']
        for category in sorted(set(or_bench['category']))
    }

    diversity = numpy.array([ lex_div.evaluate(prompts) for prompts in or_bench.values() ])
    print("[>] Properties of OR-Bench (or-bench-hard-1k)")
    print("    Number of Concepts     :", len(or_bench))
    print("    Number of Questions    :", sum(len(qdata) for qdata in or_bench.values()))
    print('    Lexical Diversity (TTR):', f"{diversity.mean():.2%}")

if __name__ == "__main__":
    datasets.disable_progress_bars()
    main()