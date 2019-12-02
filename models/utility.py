import six
from tqdm import tqdm


def create_progressbar(end, desc='', stride=1, start=0):
    return tqdm(six.moves.range(int(start), int(end), int(stride)), desc=desc, leave=False)


