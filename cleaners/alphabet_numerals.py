import re

alphabet = {
    'en': 'abcdefghijklmnopqrstuvwxyz\n',
    'fi': 'abcdefghijklmnopqrstuvwxyzåäö\n',
    'vi': 'abcdeghiklmnopqrstuvxyăâđêôơư\náàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ',
    'tr': 'aeıioöuübcçdfgğhjklmnprsştvyz\n',
    'he': 'אבגדהוזחטיכלמנסעפצקרשׂשׁתםן\nךףץְֱֲִֵֶַָֹֻּ'
}

numerals = {
    'en': {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    },
    'fi': {
        '0': 'nolla',
        '1': 'yksi',
        '2': 'kaksi',
        '3': 'kolme',
        '4': 'neljä',
        '5': 'viisi',
        '6': 'kuusi',
        '7': 'seitsemän',
        '8': 'kahdeksan',
        '9': 'yhdeksän',
    },
    'vi': {
        '0': 'không',
        '1': 'một',
        '2': 'hai',
        '3': 'ba',
        '4': 'bốn',
        '5': 'năm',
        '6': 'sáu',
        '7': 'bảy',
        '8': 'tám',
        '9': 'chín',
    },
    'tr': {
        '0': 'sıfır',
        '1': 'bir',
        '2': 'iki',
        '3': 'üç',
        '4': 'dört',
        '5': 'beş',
        '6': 'altı',
        '7': 'yedi',
        '8': 'sekiz',
        '9': 'dokuz',
    },
    'he': {
        '0': 'אֶפֶס',
        '1': 'אֶחָד',
        '2': 'שְׁנַיִם',
        '3': 'שְׁלֹשָׁה',
        '4': 'אַרְבָּעָה',
        '5': 'חֲמִשָּׁה',
        '6': 'שִׁשָּׁה',
        '7': 'שִׁבְעָה',
        '8': 'שְׁמוֹנָה',
        '9': 'תִּשְׁעָה',
    },
}


def spellout_digits(text, lang):
    assert lang in numerals
    for n, num in numerals[lang].items():
        text = text.replace(n, ' ' + num + ' ')

    return text


def keep_whitelist(text, lang):
    assert lang in alphabet
    text = re.sub(f'[^{alphabet[lang]}]+', ' ', text).strip()
    return text
