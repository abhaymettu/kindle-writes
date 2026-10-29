"""Fetch Pride and Prejudice (Gutenberg #1342) and write the training text.

Strips the Project Gutenberg header and license, maps non-ASCII characters to ASCII so
the byte-level model sees one byte per character, and drops the layout: [Illustration]
blocks, line indentation and blank-line runs. With the layout kept, a 300-step model
spends most of a page on runs of spaces copied from the centred captions. The text is
not vendored.

    python3 trainer/prepare_book.py data/pride.txt
"""

import re
import sys
import unicodedata
import urllib.request

URL = 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt'
START = '*** START OF THE PROJECT GUTENBERG EBOOK'
END = '*** END OF THE PROJECT GUTENBERG EBOOK'
MAP = {'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2014': '--', '\u2013': '-',
       '\u00b7': '', '\ufeff': '', '\u00a0': ' '}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'data/pride.txt'
    raw = urllib.request.urlopen(URL, timeout=60).read().decode('utf-8')
    body = raw[raw.index('\n', raw.index(START)) + 1:raw.index(END)]
    body = body.replace('\r\n', '\n').strip() + '\n'
    text = ''.join(MAP.get(ch, ch) for ch in body)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\[Illustration[^\]]*\]\n?', '', text)
    text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    with open(out, 'w', newline='\n') as f:
        f.write(text)
    print(f'{out}: {len(text):,} chars, {len(set(text))} distinct')


if __name__ == '__main__':
    main()
