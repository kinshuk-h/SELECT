import os
import random
import hashlib
import dataclasses

import datasets

from ..utils import io

def generate_id(text):
    """ Returns the MD5 digest for a string, to use as an ID """
    return hashlib.md5(text.encode()).hexdigest()

def sample_queries(dataset, concepts, num_queries = 10, return_concepts=False):
    all_queries = [ (query, concept) for concept in concepts for query in dataset.deepget((*concept, 'queries')) ]
    sampled_queries = random.sample(all_queries, k=num_queries) if len(all_queries) > num_queries else all_queries
    return sampled_queries if return_concepts else [ q[0] for q in sampled_queries ]

def level_traverse(taxonomy, visit_fn=None):
    """ Performs BFS traversal on a taxonomy, collecting level-wide information per node.

    Args:
        taxonomy (dict): Taxonomy tree. Each field must list children under a dict keyed as 'children'.
        visit_fn (callable, optional): Callable to invoke during visit of a node.
            Must accept the node, corresponding data and the level at which the node is found, as parameters.
            Defaults to None.
            The implementation supports live updates, so visit_fn can also modify the node data (if required),
            which impacts the taxonomy being traversed.

    Returns:
        tuple[dict, dict]: Tuple of a sibling map (siblings per node) and parent map (parent per node).
    """

    parent_map, sibling_map = {}, {}
    queue = [ ("schema:Thing", { "name": "Everything", "children": taxonomy }, 0) ]
    while len(queue) > 0:
        node, node_data, level = queue[0]
        queue = queue[1:]

        if visit_fn: visit_fn(node, node_data, level)

        children = set(node_data['children'].keys())

        if node != "schema:Thing":
            for child_node in children:
                parent_map[child_node] = node

        for child_node, child_node_data in node_data['children'].items():
            sibling_map[child_node] = list(children - set((child_node, )))
            queue.append((child_node, child_node_data, level+1))

    return sibling_map, parent_map

def concept_iterator(data_collation, compose=False):
    if compose:
        for rel, rel_data in data_collation.items():
            for concept, c_data in rel_data.items():
                yield (rel, concept), c_data
    else:
        for concept, c_data in data_collation.items():
            yield (concept, ), c_data

# ==================================================

def get_dataset(dataset_name: str, dtype='atom'):
    ref = dataset_name.upper()
    dtype      = '' if dtype == 'atom' else f'.{dtype}'
    taxon_type = '_plus' if os.path.exists(f"data/{ref}/taxonomy_plus{dtype}.json") else ''
    taxonomy   = io.Record.load(f"data/{ref}/taxonomy{taxon_type}{dtype}.json")
    return io.Record.load(f"data/{ref}/data{dtype}.json"), taxonomy

def get_train_dataset(dataset_name: str, dtype='atom'):
    ref = dataset_name.upper()
    dtype = '' if dtype == 'atom' else f'.{dtype}'
    return io.Record.load_if(f"data/{ref}/data.train{dtype}.json")

def get_eval_dataset(random_state=20240603):
    alpaca = datasets.load_dataset("tatsu-lab/alpaca")
    return alpaca['train'].train_test_split(test_size=500, shuffle=True, seed=random_state)['test']

def load_example_cache(dataset_name: str, dtype='atom'):
    ref = dataset_name.upper()
    dtype = '' if dtype == 'atom' else f'.{dtype}'
    return io.Record.load_if(f"data/{ref}/example_cache{dtype}.json")

# ==================================================

def search_taxonomy(taxonomy, target_nodes):
    """ Invokes a tree-search (using BFS) to find paths to target nodes and their corresponding data.

    Args:
        taxonomy (dict): Taxonomy tree.
        target_nodes (list[str]|str): Node(s) to search for, by ids.

    Returns:
        tuple[dict, dict]: Tuple of node data per node to be searched, and
            a dictionary of corresponding paths to nodes in the tree.
    """

    if isinstance(target_nodes, str): target_nodes = [ target_nodes ]
    target_node_data  = { node: None for node in target_nodes }
    target_node_paths = { node: None for node in target_nodes }

    def record_target_node(node, node_data, level):
        nonlocal target_node_data, target_nodes
        if node in target_nodes: target_node_data[node] = node_data

    _, parent_map = level_traverse(taxonomy, visit_fn=record_target_node)

    for node, node_data in target_node_data.items():
        if not node_data: continue
        node_path, par_node = [ node ], node
        while par_node in parent_map:
            node_path.append(parent_map[par_node])
            par_node = node_path[-1]
        target_node_paths[node] = node_path[::-1]

    return target_node_data, target_node_paths

def get_traversal_variables(taxonomy: dict | io.Record, compose_mode: bool, visit_fn=None):
    if compose_mode:
        sibling_map, parent_map = {}, {}
        node_data  , node_paths = {}, {}

        for relation, rel_taxonomy in taxonomy.items():
            if visit_fn:
                def custom_visit_fn_impl(node, data, level):
                    visit_fn((relation, node), data, level)
            else:
                custom_visit_fn_impl = None

            sibling_map[relation], parent_map[relation] = level_traverse (rel_taxonomy, custom_visit_fn_impl)
            node_data[relation]  , node_paths[relation] = search_taxonomy(
                rel_taxonomy, list(parent_map[relation]) + list(rel_taxonomy)
            )

    else:
        if visit_fn:
            def custom_visit_fn_impl(node, data, level):
                visit_fn((node, ), data, level)
        else:
            custom_visit_fn_impl = None

        sibling_map, parent_map = level_traverse (taxonomy, custom_visit_fn_impl)
        node_data  , node_paths = search_taxonomy(
            taxonomy, list(parent_map) + list(taxonomy)
        )

    return dict(
        sibling_map = io.Record(sibling_map),
        parent_map  = parent_map,
        node_data   = io.Record(node_data),
        node_paths  = node_paths
    )

@dataclasses.dataclass
class DatasetState:
    """ Class to jointly hold state information about a dataset """
    name         : str
    dtype        : str
    taxonomy     : dict|io.Record
    dataset      : dict|io.Record
    train_dataset: dict|io.Record|None
    example_cache: dict|io.Record|None
    compose      : bool

    @staticmethod
    def from_name(name, dataset_dtype='atom'):
        dataset, taxonomy = get_dataset       (name, dataset_dtype)
        example_cache     = load_example_cache(name, dataset_dtype)
        train_dataset     = get_train_dataset (name, dataset_dtype)

        return DatasetState(
            name, dataset_dtype, taxonomy,
            dataset, train_dataset, example_cache,
            compose = dataset_dtype == 'compose'
        )