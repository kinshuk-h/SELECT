import random
import hashlib

from ..utils import io

STRINGS          = io.read_yaml('templates/strings.yaml')
PROMPTS          = io.read_yaml('templates/prompts.yaml')
APPROACH_CONFIGS = io.read_yaml("config/approaches.yaml")

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