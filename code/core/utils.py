import os


def print_tree(start_path: str, padding: str = '', print_files: bool = True) -> None:
    """
    Print the directory tree starting at start_path.

    :param start_path: The directory to start the tree from.
    :param padding: The padding to use before each line (for indentation).
    :param print_files: Whether to print files as well as directories.
    """
    print(padding[:-1] + '+--' + os.path.basename(start_path) + '/')
    padding = padding + ' '
    try:
        listing = os.listdir(start_path)
    except:
        return

    for index, entry in enumerate(listing):
        if os.path.isdir(os.path.join(start_path, entry)):
            print_tree(os.path.join(start_path, entry), padding + '| ', print_files)
        elif print_files:
            print(padding + '|--', entry)

        if index == len(listing) - 1:
            print(padding[:-1] + ' `--  ', os.path.basename(start_path), end='\n\n')


def summarize_file(path):
    with open(path) as fl:
        lines = list(fl)
        print(len(lines), "lines")
        for i, ln in enumerate(lines):
            if i > 10:
                break
            print(i, "  ", len(ln.split()), ln, "\n")
