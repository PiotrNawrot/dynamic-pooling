from homoglyphs import normalise_homoglyphs
from alphabet_numerals import spellout_digits, keep_whitelist
from utils import wiki40b_markers, change_unknowns, collapse_whitespace, collapse_unknowns
from clean import text8_cleaner, soft_cleaner

txt = 'aꝸbcd\nasäd\nxä'
lang = 'en'

assert keep_whitelist(txt, lang) == 'a bcd\nas d\nx'
assert spellout_digits(txt, lang) == txt
assert spellout_digits('1\n', lang) == ' one \n'
assert normalise_homoglyphs('ꮓ ｚ\n') == 'z z\n'
assert collapse_whitespace('  aaa  bbb \n ') == ' aaa bbb \n '
assert change_unknowns('paa', ['a', 'b']) == 'p\x8e\x8e'
assert collapse_unknowns(change_unknowns('aaapaa', ['a', 'b'])) == '\x8ep\x8e'
assert wiki40b_markers('\n\n_START_ARTICLE_\n\n', mode='keep') == '\n \u008A \n'
assert wiki40b_markers('\n\n_START_ARTICLE_\n\n', mode='remove') == '\n \n'

# it's better to first normalise homoglyphs then lower
tmp = 'ΖᏃℤ'
assert normalise_homoglyphs(tmp).lower() == 'zzz'
assert normalise_homoglyphs(tmp.lower()) == 'ζzZ'

with open('./data/wiki40b/en/test.txt') as file:
    txt = file.read()

soft_clean = soft_cleaner(txt, 'en')
text8_clean = text8_cleaner(txt, 'en')

tgt_soft = """\x8a 1882 prince edward island general election \x8c the 1882 prince edward island election was held on may 8, 1882 to elect members of the house of assembly of the province of prince edward island, canada. it was won by the conservative party. \x8d the election is currently listed on the website of"""

assert soft_clean[:291] == tgt_soft
assert '\n' in soft_clean[:1000]

tgt_text8 = """one eight eight two prince edward island general election the one eight eight two prince edward island election was held on may eight one eight eight two to elect members of the house of assembly of the province of prince edward island canada it was """

assert text8_clean[:250] == tgt_text8
assert '\n' in text8_clean[:1000]
