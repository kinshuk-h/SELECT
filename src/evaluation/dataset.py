import random
import hashlib
import pathlib
import dataclasses

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
        node_paths  = io.Record(node_paths)
    )

@dataclasses.dataclass
class DatasetState:
    """ Class to jointly hold state information about a dataset """
    name            : str
    root            : pathlib.Path
    compose         : bool                 = False
    visit_fn        : 'callable'           = None
    __aux_variables : dict[dict|io.Record] = dataclasses.field(default_factory=lambda: {})

    def __load_traversal_vars(self):
        self.__aux_variables.update(get_traversal_variables(self.taxonomy, self.compose, self.visit_fn))

    def __get_var_path(self, variable):
        path_prefix = self.root / self.name.upper() / self.dtype
        if variable == 'dataset': return path_prefix / "data.json"
        elif variable == 'train_dataset': return path_prefix / "data.train.json"
        elif variable == 'example_cache': return path_prefix / "example_cache.json"
        elif (taxon_plus := path_prefix / "taxonomy_plus.json").exists(): return taxon_plus
        else: return path_prefix / "taxonomy.json"

    def __iterate(self, variable='dataset'):
        yield from concept_iterator(getattr(self, variable), self.compose)

    def keys(self, variable='dataset'):
        for key, _ in self.__iterate(variable): yield key

    def values(self, variable='dataset'):
        for _, value in self.__iterate(variable): yield value

    def items(self, variable='dataset'):
        yield from self.__iterate(variable)

    def lazy_loaded(_property):
        def property_impl(self):
            if (variable := _property.__name__) not in self.__aux_variables:
                if variable.endswith('_map') or variable.startswith('node_'): self.__load_traversal_vars()
                else: self.__aux_variables[variable] = io.Record.load_if(self.__get_var_path(variable))
            return _property(self)
        property_impl.__name__ == _property.__name__
        return property_impl

    def set_visitor(self, visit_fn=None):
        self.visit_fn = visit_fn
        for prop in ( 'sibling_map', 'parents_map', 'node_data', 'node_paths' ):
            self.__aux_variables.pop(prop, None)

    @property
    def dtype(self):
        return 'compositions' if self.compose else 'atoms'

    @property
    @lazy_loaded
    def dataset(self) -> io.Record:
        return self.__aux_variables['dataset']

    @property
    @lazy_loaded
    def train_dataset(self) -> io.Record:
        return self.__aux_variables['train_dataset']

    @property
    @lazy_loaded
    def taxonomy(self) -> io.Record:
        return self.__aux_variables['taxonomy']

    @property
    @lazy_loaded
    def example_cache(self) -> io.Record:
        return self.__aux_variables['example_cache']

    @property
    @lazy_loaded
    def node_data(self) -> io.Record:
        return self.__aux_variables['node_data']

    @property
    @lazy_loaded
    def node_paths(self) -> io.Record:
        return self.__aux_variables['node_paths']

    @property
    @lazy_loaded
    def sibling_map(self) -> io.Record:
        return self.__aux_variables['sibling_map']

    @property
    @lazy_loaded
    def parent_map(self) -> dict:
        return self.__aux_variables['parent_map']