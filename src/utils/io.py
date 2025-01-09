"""

    io
    ~~

    Utilities for I/O operations, and
    wrappers over base types for JSON serialization.

"""

import os
import re
import gzip
import json
import yaml
import textwrap

try:
    import msgspec
    def __json_deserialize(filep): return msgspec.json.decode(filep.read())
except ImportError:
    def __json_deserialize(filep): return json.load(filep)

# = General ============================================================================================================

def pathsafe(filename):
    """ Returns a santized, path-safe version of a filename. """
    return re.sub(r'[:/\\|*]', '-', re.sub(r'[?\"<>]', '', filename))

def wrap(text: str, width=100, indent=0, indent_unit=' '):
    """ Newline-aware textwrap wrapping.

    Args:
        text (str): Text to wrap.
        width (int, optional): Maximum width. Defaults to 100.
        indent (int, optional): Amount of global indent. Defaults to 0.
        indent_unit (str, optional): String to use for indents. Defaults to ' '.

    Returns:
        str: Text wrapped to width constraints, respecting newlines.
    """
    _indent = '' if not indent else (indent_unit * indent)
    return '\n'.join([
        '\n'.join(_indent + line for line in textwrap.fill(segment, width).splitlines())
        for segment in text.splitlines()
    ])

# = Structured I/O =====================================================================================================

def read_yaml(path):
    """ Loads a YAML file. """
    with open(path, 'r', encoding='utf-8') as ifile:
        return yaml.load(ifile, Loader=yaml.SafeLoader)

def write_yaml(path, data, indent=2, ensure_ascii=False, **kwargs):
    """ Saves an object to a YAML file, with some default options. """
    with open(path, 'w', encoding='utf-8') as ofile:
        yaml.dump(
            data, ofile, Dumper=yaml.SafeDumper, indent=indent,
            allow_unicode=not ensure_ascii, **kwargs
        )

def read_json(path, compressed=None):
    """ Loads a JSON file.

    Args:
        path (str | pathlib.Path): Path to load the file from.
        compressed (bool, optional): Whether the file is GZIP compressed.
            Defaults to None, where this is assumed based on file extension.

    Returns:
        any: Data read from the JSON file.
    """
    if compressed is None: compressed = str(path).endswith(".gz")
    io_function = gzip.open if compressed else open

    with io_function(path, 'rt', encoding='utf-8') as ifile:
        return __json_deserialize(ifile)

def write_json(path, data, compress=None, **kwargs):
    """ Saves an object to a JSON file, with some default options.

    Args:
        path (str): Path to write the file to.
        data (any): Data to serialize as JSON.
        compress (bool, optional): Whether to GZIP compress the file.
            Defaults to None, where this is assumed based on file extension.
    """
    if compress is None: compress = str(path).endswith(".gz")
    io_function = gzip.open if compress else open

    __serialize_kwargs = dict(indent=2, ensure_ascii=False)
    __serialize_kwargs.update(**kwargs)

    with io_function(path, 'wt', encoding='utf-8') as ofile:
        json.dump(data, ofile, **__serialize_kwargs)

def read_jsonl(path, compressed=None):
    """ Loads a JSON file.

    Args:
        path (str | pathlib.Path): Path to load the file from.
        compressed (bool, optional): Whether the file is GZIP compressed.
            Defaults to None, where this is assumed based on file extension.

    Returns:
        any: Data read from the JSON file.
    """

    if compressed is None: compressed = str(path).endswith(".gz")
    io_function = gzip.open if compressed else open

    with io_function(path, 'rt', encoding='utf-8') as ifile:
        return [ json.loads(record) for record in ifile ]

def write_jsonl(path, data, compress=None):
    """ Loads a JSON file.

    Args:
        path (str | pathlib.Path): Path to load the file from.
        compressed (bool, optional): Whether the file is GZIP compressed.
            Defaults to None, where this is assumed based on file extension.

    Returns:
        any: Data read from the JSON file.
    """

    if compress is None: compress = str(path).endswith(".gz")
    io_function = gzip.open if compress else open

    with io_function(path, 'wt', encoding='utf-8') as ofile:
        for entry in data: ofile.write(json.dumps(entry) + '\n')

def read_lines(path, max=-1):
    """ Read lines off a text file """
    with open(path, 'r', encoding='utf-8') as ifile:
        return ifile.readlines(max)

def write_lines(path, lines):
    """ Write lines to a text file """
    with open(path, 'w', encoding='utf-8') as ofile:
        ofile.writelines(lines)

# = Serialization ======================================================================================================

class Record(dict):
    """ Wrapper over dict, to support deep key get and set operations, along with JSON serialization. """

    def __init__(self, *args, **kwargs):
        """ Initialize a Record object """
        dict.__init__(self, *args, **kwargs)
        for key, value in self.items():
            self.__setitem__(key, value)

    @staticmethod
    def indexable(source, key):
        """ Multi-utility wrapper to determine if a key is present in source.
            Works for both list and dict types.

        Args:
            source (list|dict|tuple|Record): Source to check
            key (int|str|bytes): Key to search for.

        Returns:
            bool: True if key is found, else False
        """
        if isinstance(source, (dict, Record)):
            return key in source and source[key] is not None
        elif isinstance(source, (list, tuple)):
            return len(source) > key
        else:
            return False

    @staticmethod
    def __convert(o):
        """ Recursively convert `dict` objects in `dict`, `list`, `set`, and `tuple` objects to `Record` objects. """
        if isinstance(o, dict)   : o = Record(o)
        elif isinstance(o, list) : o = list(Record.__convert(v) for v in o)
        elif isinstance(o, set)  : o = set(Record.__convert(v) for v in o)
        elif isinstance(o, tuple): o = tuple(Record.__convert(v) for v in o)
        return o

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no key/attribute '{k}'")

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, self.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no key/attribute '{k}'")

    @staticmethod
    def load(path):
        """ De-serializes a record object from a JSON file. """
        return Record(read_json(path))

    @staticmethod
    def load_if(path):
        """ De-serializes a record object from a JSON file, if the file exists.
            Returns an empty record otherwise. """
        if os.path.exists(path):
            return Record(read_json(path))
        else:
            return Record()

    def __repr__(self) -> str:
        """ Representation of the Record object. """
        return f"Record({super().__repr__()})"

    def save(self, path):
        """ Serializes a record object to the specified path. """
        write_json(path, self)

    def deepget(self, deepkey, default=None):
        """ Performs a deepkey retrieval: Resolves the value at the
                nested key if available, else falls back to default.

        Args:
            deepkey (str|tuple): Nested key, either a string separated by '.', or a tuple.
            default (any, optional): Default value in case of failure in retrieval. Defaults to None.

        Returns:
            any: Value saved at the nested key or the default.

        Example:
        >>> record = Record({ 'a': { 'b': { 'c': "Hello" } } })
        # Equivalent to record.get('a', { 'b': { 'c': None } }).get('b', { 'c': None }).get('c', None)
        >>> record.deepget("a.b.c")
        "Hello"
        >>> record.deepget(('a', 'e', 'c'))
        None
        """
        retval = self
        for key in (deepkey if isinstance(deepkey, (list, tuple)) else deepkey.split('.')):
            if not self.indexable(retval, key): return default
            retval = retval[key]
        return retval

    def deepset(self, deepkey, value):
        """ Performs a deepkey set operation: Sets the value at any arbitrary nested key,
            automatically creating parent dictionaries as required.

        Args:
            deepkey (str|tuple): Nested key, either a string separated by '.', or a tuple.
            value (any): Value to set at the nested key. Parents are automatically created.

        Example:
        >>> record = Record()
        >>> record.deepset("a.b.c", "Hello")
        >>> record
        Record({ 'a': { 'b': { 'c': "Hello" } } })
        >>> record.deepset(('a', 'e', 'c'), "Bye")
        >>> record
        Record({ 'a': { 'b': { 'c': "Hello" }, 'e': { 'c': "Bye" } } })
        """
        keys = deepkey if isinstance(deepkey, (list, tuple)) else deepkey.split('.')
        ref = self
        for i, key in enumerate(keys):
            if i == len(keys)-1: break
            if not self.indexable(ref, key):
                ref[key] = {}
            ref = ref[key]
        ref[keys[-1]] = value

    def deepref(self, deepkey, value):
        """ Performs a deepkey reference operation: retrieves the reference at any arbitrary nested key,
            automatically creating parent dictionaries as required. If the reference doesn't exist,
            a default value as specified is used. Useful for nested object and iterables.

        Args:
            deepkey (str|tuple): Nested key, either a string separated by '.', or a tuple.
            value (any): Default value to use for the nested key, if no reference exists.

        Example:
        >>> record = Record()
        >>> data = record.deepref("a.b.c", [])
        >>> record
        Record({ 'a': { 'b': { 'c': [] } } })
        >>> data.append(1)
        >>> record
        Record({ 'a': { 'b': { 'c': [ 1 ] })

        """
        keys = deepkey if isinstance(deepkey, (list, tuple)) else deepkey.split('.')
        ref = self
        for i, key in enumerate(keys):
            if i == len(keys)-1: break
            if not self.indexable(ref, key): ref[key] = {}
            ref = ref[key]
        if keys[-1] not in ref: ref[keys[-1]] = value
        return ref[keys[-1]]

# ======================================================================================================================