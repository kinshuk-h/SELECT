"""

    utils
    ~~~~~

    Model inference wrapper utilities, such as default prompts and functions.

"""

def contains(text: str, words):
    """ Checks if a string contains any of the given words in an iterable. """
    return any(word in text for word in words)

def is_conversation(prompt: list|str):
    """ Detects if a (batched) prompt(s) corresponds to conversation/chat instances. """
    return isinstance(prompt, list) and (
        isinstance(prompt[0], dict) or (
            isinstance(prompt[0], list) and isinstance(prompt[0][0], dict)
        )
    )