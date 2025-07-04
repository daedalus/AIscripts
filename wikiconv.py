# translation from https://mattmahoney.net/dc/textdata.html

import sys
import re

def spell_digits(text):
    return (text.replace('0', ' zero ')
                .replace('1', ' one ')
                .replace('2', ' two ')
                .replace('3', ' three ')
                .replace('4', ' four ')
                .replace('5', ' five ')
                .replace('6', ' six ')
                .replace('7', ' seven ')
                .replace('8', ' eight ')
                .replace('9', ' nine '))

text_mode = False

for line in sys.stdin:
    if '<text ' in line:
        text_mode = True
    if re.search(r'#redirect', line, re.IGNORECASE):
        text_mode = False
    if text_mode:
        if '</text>' in line:
            text_mode = False
        
        # Remove XML tags
        line = re.sub(r'<.*?>', '', line)
        
        # Decode URL encoded chars
        line = line.replace('&amp;', '&')
        line = line.replace('&lt;', '<')
        line = line.replace('&gt;', '>')
        
        # Remove references
        line = re.sub(r'<ref[^<]*?</ref>', '', line)
        
        # Remove XHTML tags
        line = re.sub(r'<[^>]*>', '', line)
        
        # Remove normal URLs, preserve visible text
        line = re.sub(r'\[http:[^\]\s]*', '[', line)
        
        # Remove image link details, preserve caption
        line = re.sub(r'\|thumb', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\|left', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\|right', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\|\d+px', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\[\[image:[^\[\]]*\|', '', line, flags=re.IGNORECASE)
        
        # Show categories without markup
        line = re.sub(r'\[\[category:([^|\]]*)[^\]]*\]\]', r'[[\1]]', line, flags=re.IGNORECASE)
        
        # Remove links to other languages
        line = re.sub(r'\[\[[a-z\-]*:[^\]]*\]\]', '', line)
        
        # Remove wiki URL, preserve visible text
        line = re.sub(r'\[\[[^\|\]]*\|', '[[', line)
        
        # Remove templates ({{...}}, {...})
        line = re.sub(r'\{\{[^}]*\}\}', '', line)
        line = re.sub(r'\{[^}]*\}', '', line)
        
        # Remove [ and ]
        line = line.replace('[', '')
        line = line.replace(']', '')
        
        # Remove remaining URL encoded entities
        line = re.sub(r'&[^;]*;', ' ', line)
        
        # Convert to lowercase and spell out digits
        line = ' ' + line + ' '
        line = line.lower()
        line = spell_digits(line)
        
        # Convert all non-lowercase a-z to space, and squeeze spaces
        line = re.sub(r'[^a-z]', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        
        print(line)

