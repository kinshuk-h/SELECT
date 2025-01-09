"""

    formatting
    ~~~~~~~~~~

    Utility functions for formatting data.

"""

import string

# ========================================================================

__TIME_UNITS__  = [ 'ms', 's', 'm', 'h', 'd', 'w', 'y' ]
__TIME_RATIOS__ = ( 1000, 60, 60, 24, 7, 52 )

def format_time(time_in_s, ret_type='iter', ratio=0.7):
    """ Format seconds to a time string.

    Args:
        time_in_s (int): Time elapsed in seconds.
        ret_type (str, optional): Return format mode. Defaults to 'iter'.
        ratio (float, optional): Ratio to determine promotion to next unit. Defaults to 0.7.

    Returns:
        str: Formatted time string.
    """
    time_in_ms = time_in_s * 1e3
    if ret_type == 'iter':
        index = 0
        while index < len(__TIME_RATIOS__) and time_in_ms > (ratio * __TIME_RATIOS__[index]):
            time_in_ms /= __TIME_RATIOS__[index]
            index += 1
        fmtd_time = f"{time_in_ms:.1f}{__TIME_UNITS__[index]}"
    else:
        index = 0
        fmtd_time = ""
        while index < len(__TIME_RATIOS__) and time_in_ms > (ratio * __TIME_RATIOS__[index]):
            mod_time = time_in_ms % __TIME_RATIOS__[index]
            if mod_time != 0:
                fmtd_time = f"{int(mod_time)}{__TIME_UNITS__[index]}" + fmtd_time
            time_in_ms //= __TIME_RATIOS__[index]
            index += 1
        if time_in_ms != 0:
            fmtd_time = f"{int(time_in_ms)}{__TIME_UNITS__[index]}" + fmtd_time
    return fmtd_time

# ========================================================================

__SIZES__ = " KMGTPEZY"

def format_size(size_bytes, use_si=False, ratio=0.7):
    """ Formats size in bytes to a size string.

    Args:
        size_bytes (int): Size to format, in bytes.
        use_si (bool, optional): If True, uses SI ratios (1K = 1000). Defaults to False.
        ratio (float, optional): Ratio to determine promotion to the next unit. Defaults to 0.7.

    Returns:
        str: Formatted size string.
    """
    factor = 1000 if use_si else 1024
    idx, size = 0, size_bytes
    while size >= ratio * factor:
        idx += 1
        size /= factor
    return f"{size:.3f} {__SIZES__[idx].strip()}{'i' if not use_si and idx > 0 else ''}B"

# ========================================================================

__FORMATTER__ = string.Formatter()

def is_number(x):
    """ Checks if x is a number. """
    try: int(x); return True
    except: return False

def format_prompt(prompt_template, *args, **kwargs):
    """ Formats a prompt containing format specifiers partially with available parameters.

    Args:
        prompt_template (str): Prompt template to format (completely or partially.)

    Returns:
        str: Formatted prompt based on give parameters. May still have non-substituted terms.
    """
    if isinstance(prompt_template, list):
        new_prompt_template = []
        for turn in prompt_template:
            new_turn = { **turn }
            new_turn['content'] = format_prompt(new_turn['content'], *args, **kwargs)
            new_prompt_template.append(new_turn)
        prompt_template = new_prompt_template
    else:
        parsed_fields = [ item for _, item, _, _ in __FORMATTER__.parse(prompt_template) if item ]

        repl_dict = { item: f"{{{item}}}" for item in parsed_fields if not is_number(item) }
        repl_dict.update(**{ key: value for key, value in kwargs.items() if key in repl_dict })

        repl_args = [ f"{{{item}}}" for item in parsed_fields if is_number(item) ]
        repl_args[:len(args)] = args

        prompt_template = prompt_template.format(*repl_args, **repl_dict)
    return prompt_template

# ========================================================================