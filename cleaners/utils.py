from multiprocessing import Pool
from functools import partial
import re

unk = '\u008E'


def change_unks(disallowed_symbols, txt):
    assert unk not in txt

    for char in disallowed_symbols:
        txt = txt.replace(char, unk)

    return txt


def change_unknowns(text, disallowed_symbols, n_chunks=32):
    chunk_len = (len(text) + n_chunks - 1) // n_chunks
    splitted_text = [text[i:i + chunk_len] for i in range(0, len(text), chunk_len)]

    partial_change_unks = partial(change_unks, disallowed_symbols)

    with Pool(n_chunks) as p:
        text = ''.join(p.imap(partial_change_unks, splitted_text))

    return text


def wiki40b_markers(text, mode):
    markers = ['\n_START_ARTICLE_\n', '\n_START_SECTION_\n', '\n_START_PARAGRAPH_\n', '_NEWLINE_']
    specials = ['\u008A', '\u008B', '\u008C', '\u008D']

    assert mode in ['keep', 'remove']

    # Make sure the special symbols are not in the dataset as we'd overload
    # them otherwise
    for marker, special in zip(markers, specials):
        if mode == 'keep':
            assert special not in text
            text = text.replace(marker, f' {special} ')
        elif mode == 'remove':
            text = text.replace(marker, '\n')

    return text


def collapse_whitespace(text):
    # It doesn't collapse/remove endlines, only whitespaces
    text = re.sub(re.compile(r' +'), ' ', text)
    text = re.sub(re.compile(r'( *[\n]+ *)+'), '\n', text)
    return text


def collapse_unknowns(text):
    return re.sub(re.compile(fr'{unk}+'), unk, text)
