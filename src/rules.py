"""
Linguistic rules shared by the synthetic error generator and by the feedback engine:
the same knowledge is used to create training errors and to explain them to the writer.
"""

# Typing errors: each key mapped to its neighbours on the keyboard
KEYBOARD_NEIGHBOURS = {
    'q': ['w', 'a'],                     'w': ['q', 'e', 'a', 's'],
    'e': ['w', 'r', 's', 'd'],           'r': ['e', 't', 'd', 'f'],
    't': ['r', 'y', 'f', 'g'],           'y': ['t', 'u', 'g', 'h', 'j'],
    'u': ['y', 'i', 'h', 'j'],           'i': ['u', 'o', 'j', 'k'],
    'o': ['i', 'p', 'k', 'l'],           'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'],           's': ['w', 'e', 'a', 'd', 'z', 'x'],
    'd': ['e', 'r', 's', 'f', 'x', 'c'], 'f': ['r', 't', 'd', 'g', 'c', 'v'],
    'g': ['t', 'y', 'f', 'h', 'v', 'b'], 'h': ['y', 'u', 'g', 'j', 'b', 'n'],
    'j': ['u', 'i', 'h', 'k', 'n', 'm'], 'k': ['i', 'o', 'j', 'l', 'm'],
    'l': ['i', 'o', 'p', 'k'],           'z': ['a', 's', 'x'],
    'x': ['s', 'd', 'z', 'c'],           'c': ['d', 'f', 'x', 'v'],
    'v': ['f', 'g', 'c', 'b'],           'b': ['g', 'h', 'v', 'n'],
    'n': ['h', 'j', 'b', 'm'],           'm': ['j', 'k', 'n'],

    'à': ['a'],                          'è': ['e'],
    'é': ['e'],                          'ì': ['i'],
    'ò': ['o'],                          'ù': ['u'],
}

# Confusion sets based on the Sardinian graphematic repertoire.
# Each correct grapheme is mapped to the spellings it is commonly mistaken for.
GRAPHEMATIC_RULES = {
    # Degemination / hypercorrection (and lenition)
    'bb': ['b'],        'b': ['bb', 'v'],       'v':   ['b'],
    'cc': ['c'],
    'dd': ['d'],        'd': ['r', 't', 'dd'],
    'ff': ['f'],        'f': ['ff', 'v'],
    'gg': ['g'],        'g': ['gh', 'gg'],
    'll': ['l'],
    'mm': ['m'],
    'nn': ['n'],
    'pp': ['p'],        'p': ['pp', 'b'],
    'rr': ['r'],        'r': ['l', 'rr'],
    'ss': ['s'],        's': ['ss', 'z'],
    'tt': ['t'],        't': ['d', 'tt', 'th'], 'th':  ['t', 'z', 's'],

    # Nasals and palatals
    'mb':  ['mm'],        'nd':  ['nn'],
    'gi':  ['j', 'ghi'],  'gn':  ['nn', 'ni'],
    'j':   ['i', 'gi'],

    # Affricates: a major source of confusion in Sardinian spelling
    'tz': ['z', 'ss', 'c'], 'z':  ['tz', 's'],

    # Italian-Sardinian conflicts, from speech to writing
    'xe':  ['sce', 'ge', 'se'],   'xi':  ['sci', 'gi', 'si'],
    'sce': ['xe', 'se', 'ce'],    'sci': ['xi', 'si', 'ci'],
    'ci':  ['sci', 'chi'],        'ce':  ['sce', 'che'],
    'che': ['ce', 'ke'],          'chi': ['ci', 'ki'],
    'ghe': ['ge'],                'ghi': ['gi'],
}

# Graphemes whose spelling differs between Italian and Sardinian conventions
ITALIAN_CONFLICT_GRAPHEMES = {'xe', 'xi', 'sce', 'sci', 'ce', 'ci', 'che', 'chi', 'ghe', 'ghi',
                              'ke', 'ki', 'ge', 'gi', 'se', 'si', 'gn', 'ni', 'j'}

# Sibilants and affricates
SIBILANT_GRAPHEMES = {'tz', 'z', 'th'}


def grapheme_family(correct: str, wrong: str) -> str:
    """ Groups a pair of confused graphemes into a family, used to explain the confusion. """
    shorter, longer = sorted((correct, wrong), key=len)
    if len(longer) == 2 and longer[0] == longer[1] and shorter == longer[0]:
        return "double_consonant"
    if {correct, wrong} & ITALIAN_CONFLICT_GRAPHEMES:
        return "italian_conflict"
    if {correct, wrong} & SIBILANT_GRAPHEMES:
        return "sibilant"
    return "similar_sound"
