from collections import Counter
import sys

from homoglyphs import normalise_homoglyphs
from alphabet_numerals import spellout_digits, keep_whitelist
from utils import wiki40b_markers, change_unknowns, collapse_whitespace, collapse_unknowns


def text8_cleaner(text, lang):
    text = wiki40b_markers(text, mode='remove')

    if lang != 'he':
        text = normalise_homoglyphs(text)

    text = text.lower()

    text = spellout_digits(text, lang)
    text = keep_whitelist(text, lang)

    text = collapse_whitespace(text)
    text = text.strip()
    return text


def soft_cleaner(text, lang, threshold=5, valid_test_size=int(5e6)):
    text = wiki40b_markers(text, mode='keep')
    text = text.lower()

    # This cleaner has to be applied to concatenated train/valid/test
    # Apart from replacing least frequent symbols with \unk we also want
    # to change to unks all symbols that are not in train but in valid/test.
    trainset = text[:2 * valid_test_size]
    train_counter = Counter(trainset)
    allowed_symbols = set([k for k, v in train_counter.items() if v > threshold])
    all_symbols = set(text)
    disallowed_symbols = all_symbols - allowed_symbols
    text = change_unknowns(text, disallowed_symbols=disallowed_symbols)
    text = collapse_unknowns(text)

    text = collapse_whitespace(text)
    text = text.strip()
    return text


if __name__ == "__main__":
    # Arguments
    filename = sys.argv[1]
    lang = sys.argv[2]
    cleaner = sys.argv[3]

    with open(filename) as file:
        text = file.read()

    if cleaner == 'text8':
        text = text8_cleaner(text, lang)
    elif cleaner == 'soft':
        text = soft_cleaner(text, lang)
    else:
        raise NotImplementedError

    sys.stdout.write(text)
