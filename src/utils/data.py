import os

from . import io

class Result:
    """ Wrapper over a record to maintain experiment results. """

    def __init__(self, file) -> None:
        self.results = io.Record()
        self.file    = file

    def load(self):
        return io.Record.load_if(self.file)

    def save(self):
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        self.results.save(self.file)

    def __getitem__(self, key):
        return self.results[key]

    def __setitem__(self, key, value):
        self.results[key] = value

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def deepget(self, keyset: str | list | tuple, default=None):
        return self.results.deepget(keyset, default=default)

    def deepset(self, keyset: str | list | tuple, value):
        self.results.deepset(keyset, value=value)

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def data(self):
        return self.results

class NestedListItemResult(Result):
    """ Type of result comprising of nested keys, initialized from list of lists """

    def __init__(self, file, *key_lists) -> None:
        super().__init__(file)
        self.results = self.__init_or_load(*key_lists)

    @classmethod
    def recursive_create(cls, key_lists, level=0):
        """ Recursively construct a result object from lists of keys.

        Args:
            key_lists (list[list]): List of key lists to use.
            level (int, optional): Level to insert the keys at. Defaults to 0.

        Returns:
            Record: Result object using key list hierarchy.
        """
        if level < len(key_lists):
            return io.Record({
                key: cls.recursive_create(key_lists, level+1)
                for key in key_lists[level]
            })
        else:
            return None

    @classmethod
    def recursive_update(cls, results, key_lists, level=0):
        """ Recursively updates a result object from lists of keys.

        Args:
            results (Record): result to update
            key_lists (list[list]): List of key lists to use
            level (int, optional): Level to insert the keys at. Defaults to 0.

        Returns:
            Record: Updated result object with new keys inserted.
        """
        if level < len(key_lists):
            for key in key_lists[level]:
                if key not in results:
                    results[key] = cls.recursive_create(key_lists, level+1)
                else:
                    results[key] = cls.recursive_update(results[key], key_lists, level+1)
            return results
        else:
            return results

    def __init_or_load(self, *key_lists):
        """ Creates or updates a result object, with data synced from the source file. """

        if os.path.exists(self.file):
            results = self.load()
            if len(key_lists) > 0:
                results = self.recursive_update(results, key_lists)

        else:
            results = self.recursive_create(key_lists) or io.Record()

        return results
