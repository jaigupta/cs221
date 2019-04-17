import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    import re
    if not text:
        return ''
    lines = re.split(r'\s+', text)
    if not lines:
        return ''
    lines.sort()
    return lines[-1]
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt(math.pow(loc1[0] - loc2[0], 2) + math.pow(loc1[1]-loc2[1], 2))
    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    if sentence is None or sentence == "":
        return []
    import re
    in_tokens = filter(None, re.split(r'\s+', sentence))
    next_map = {}
    for i in range(len(in_tokens)-1):
        token = in_tokens[i]
        if not token in next_map:
            next_map[token] = set()
        next_map[token].add(in_tokens[i+1])
    def recurseSol(cur, last, n, visited, result):
        if cur in visited:
            return
        visited.add(cur)
        if n == 0:
            result.append(cur)
            return
        for token in next_map.get(last, []):
            recurseSol(cur + ' ' + token, token, n-1, visited, result)
    visited = set()
    result = []
    for token in set(in_tokens):
        recurseSol(token, token, len(in_tokens)-1, visited, result)
    # print sentence
    # print result
    return result
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    keys = v1.keys() if len(v1) < len(v2) else v2.keys()
    return sum([v1[i]*v2[i] for i in keys])
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for i in set(v1.keys() + v2.keys()):
        v1[i] += scale*v2[i]
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    if text is None or text == "":
        return set()
    import re
    counter = collections.defaultdict(int)
    for token in re.split(r'\s+', text):
        counter[token]+=1
    return set([key for key, value in counter.items() if value==1])
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    if text is None or text == "":
        return 0
    n = len(text)
    if n <= 1:
        return n
    dp = [[0]*n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
        if i != n-1:
            dp[i][i+1] = (2 if text[i] == text[i+1] else 1)
    for  j in range(2, n):
        for i in range(n-j):
            dp[i][i+j] = max(
                dp[i+1][i+j],
                dp[i][i+j-1],
                dp[i+1][i+j-1] + (2 if text[i] == text[i+j] else 0))
    return dp[0][n-1]
    # END_YOUR_CODE
