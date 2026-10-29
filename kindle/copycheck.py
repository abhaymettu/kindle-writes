"""copycheck.py TEXT SAMPLE...   Longest run of words in each sample that appears verbatim in the training text.
Samples with a long run are reciting a book, not writing; they are kept out of anything shown or published."""
import re, sys

def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())

N = 6
corpus = words(open(sys.argv[1], encoding="ascii", errors="ignore").read())
grams = {tuple(corpus[i:i + N]) for i in range(len(corpus) - N + 1)}
text = " " + " ".join(corpus) + " "

for path in sys.argv[2:]:
    w = words(open(path, encoding="ascii", errors="ignore").read())
    best, i = 0, 0
    while i + N <= len(w):
        if tuple(w[i:i + N]) in grams:
            j = i + N
            while j < len(w) and (" " + " ".join(w[i:j + 1]) + " ") in text:
                j += 1
            best = max(best, j - i)
            i = j
        else:
            i += 1
    print(f"{best:3d}  {path}")
