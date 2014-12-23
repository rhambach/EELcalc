"""
Module to translate various names (unicode, LaTeX & other text) for characters to encodings in the Symbol font standard encodings.
Also, provide grace markup strings for them. 
It recognizes unicode names for the greek alphabet and most of the useful symbols in the Symbol font.

Marcus Mendenhall, Vanderbilt University, 2006
$Id: symbol_mapping.py,v 1.2 2009/04/03 00:32:07 mendenhall Exp $
"""
#a tuple of tuple of the position of the character in the standard Symbol encoding, and all aliases
_symbols=[
    (0xa0, u"\u2202", "euro"),
    (0xa1, u"\u03d2", "upshook"),
    (0xa2, u"\u02b9", "prime"),
    (0xa3, u"\u2264", "leq", "lessequal"),
    (0xa4, u"\u2044", "fraction", "fractionslash"),
    (0xa5, u"\u221E", "infinity",  "infty"), 
    (0xa6, u"\u0192", "f", "function", "fhook"),
    (0xa7, u"\u2663", "club"),
    (0xa8, u"\u2666", "diamond"),
    (0xa9, u"\u2665", "heart"),
    (0xaa, u"\u2660", "spade"),
    (0xab, u"\u2194", "leftrightarrow", "lrarrow"),
    (0xac, u"\u2190", "leftarrow", "larrow"),
    (0xad, u"\u2191", "uparrow"),
    (0xae, u"\u2192", "rightarrow", "rarrow"),
    (0xaf, u"\u2193", "downarrow"),
    
    (0xb0, u"\u00b0", "degree"),
    (0xb1, u"\u00b1", "plusminus"), 
    (0xb2, u"\u02ba", "primeprime", "doubleprime", "prime2"),
    (0xb3, u"\u2265", "geq", "greaterequal"), 
    (0xb4, u"\u00d7", "times"),
    (0xb5, u"\u221d",  "proportional", "propto"), 
    (0xb6, u"\u2202", "partial"),
    (0xb7, u"\u2022", "cdot", "bullet"), 
    (0xb8, u"\u00f7", "divide"), 
    (0xb9, u"\u2260", "notequal", "neq"), 
    (0xba, u"\u2261", "equiv", "equivalence" ), 
    (0xbb, u"\u2248", "approx", "almostequal"), 
    (0xbc, u"\u2026", "ellipsis", "3dots"), 
    (0xbd, u"\u007c",  "vertical", "solidus"), 
    (0xbe, u"\u23af", "horizontal", "longbar"),
    (0xbf, u"\u21b5",  "downleftarrow"), 
    
    (0xc0, u"\u2135", "aleph", "alef"), 
    (0xc1, u"\u2111", "script-letter-I"),
    (0xc2, u"\u211c", "script-letter-R"), 
    (0xc3, u"\u2118", "script-letter-P"),
    (0xc4, u"\u2297", "circled-times"), 
    (0xc5, u"\u2295", "circled-plus"),
    (0xc6, u"\u2205", "emptyset"),
    (0xc7, u"\u2229", "intersection"),
    (0xc8, u"\u222a", "union"),
    (0xc9, u"\u2283", "superset"),
    (0xca, u"\u2287", "superset-or-equal"),
    (0xcb, u"\u2284", "not-subset"),
    (0xcc, u"\u2282", "subset"),
    (0xcd, u"\u2286", "subset-or-equal"),
    (0xce, u"\u2208", "element"),
    (0xcf, u"\u2209", "not-element"),

    (0xd0, u"\u2220", "angle"),
    (0xd1, u"\u2207", "del", "nabla", "gradient"),
    (0xd2, u"\uf8e8", "registered-serif"),
    (0xd3, u"\uf8e9", "copyright-serif"),
    (0xd4, u"\uf8ea", "trademark-serif"),
    (0xd5, u"\u220f", "product"),
    (0xd6, u"\u221a", "sqrt", "radical", "root"),
    (0xd7, u"\u22c5", "cdot", "center-dot", "dot-operator"),
    (0xd8, u"\u00ac", "not"),
    (0xd9, u"\u2227", "logical-and", "conjunction"), 
    (0xda, u"\u2228", "logical-or", "disjunction", "alternation"),
    (0xdb, u"\u21d4", "left-right-double-arrow", "iff"),
    (0xdc, u"\u21d0", "left-double-arrow"),
    (0xdd, u"\u21d1", "up-double-arrow"),
    (0xde, u"\u21d2", "right-double-arrow", "implies"),
    (0xdf, u"\u21d3", "down-double-arrow"),

    (0xe0, u"\u25ca", "lozenge"),
    (0xe1, u"\u3008", "left-angle-bracket", "langle"),  
    (0xe2, u"\u00ae", "registered-sans"),
    (0xe3, u"\u00a9", "copyright-sans"),
    (0xe4, u"\u2122", "trademark-sans"),
    (0xe5, u"\u2211", "sum"),

    (0xf2, u"\u2228", "integral"),

]

#insert unicodes for official greek letters, so a pure unicode string with these already properly encoded will translate correctly
_greekorder='abgdezhqiklmnxoprvstufcyw' 
_greekuppernames=['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota',
        'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'finalsigma', 
        'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega']
 
#note thet @ sign is in the place of the 'final sigma' character in the standard greek alphabet mapping for the symbol font

_ugrcaps="".join([unichr(x) for x in range(0x391, 0x391+len(_greekorder))])
_symbols += zip([ord(x) for x in _greekorder.upper()], _ugrcaps)
_symbols += zip([ord(x) for x in _greekorder.upper()], _greekuppernames)

_ugrlower="".join([unichr(x) for x in range(0x3b1, 0x3b1+len(_greekorder))])
_symbols += zip([ord(x) for x in _greekorder], _ugrlower)
_greeklowernames=[x.lower() for x in _greekuppernames]
_greeklowernames[_greeklowernames.index('finalsigma')]='altpi' 
_symbols += zip([ord(x) for x in _greekorder], [x.lower() for x in _greeklowernames])

gracedict={}

for tt in _symbols:
    if tt[0] > 0x20 and tt[0] < 0x7f:
        vstr=r"\x"+chr(tt[0])+r"\f{}"
    else:
        vstr=r"\x\#{%02x}\f{}" % tt[0]
    for tag in tt[1:]:
        gracedict[tag]=vstr

_normalascii="".join([chr(i) for i in range(32,127)])
_normalucode=unicode(_normalascii)

def remove_redundant_changes(gracestring):
    """collapse out consecutive font-switching commands so that \xabc\f{}\xdef\f{} becomes \xabcdef\f{}"""
    while(1):
        xs=gracestring.find(r"\f{}\x")
        if xs<0: break
        if xs >=0:
            gracestring=gracestring[:xs]+gracestring[xs+6:]
    return gracestring
    
def translate_unicode_to_grace(ucstring):
    """take a string consisting of unicode characters for a mixture of normal characters 
    and characters which map to glyphs in Symbol and create a Grace markup string from it"""
    outstr=""
    for uc in ucstring:
        if uc in _normalucode:
            outstr+=str(uc) #better exist in ascii
        else:
            outstr+=gracedict.get(uc,"?")
    
    return remove_redundant_changes(outstr)

def format_python_to_grace(pystring):
    """take a string with %(alpha)s%(Upsilon)s type coding, and make a Grace markup string from it"""
    return remove_redundant_changes(pystring % gracedict)

if __name__=="__main__":
    # a ur"foo" string is raw unicode, iin which only \u is interpreted, so it is good for grace escapes
    print   translate_unicode_to_grace(ur"Hello\xQ\f{}\u0391\u03b1\u2227\u22c5\u03c8\u03a8\u03c9\u03a9")

    import sys
    import time
    import os
    
    import GracePlot
    
    class myGrace(GracePlot.GracePlot):
    
        def write_string(self, text="", font=0, x=0.5, y=0.5, size=1.0, just=0, color=1, coordinates="world", angle=0.0):
            
            strg="""with string
                string on
                string loctype %(coordinates)s
                string %(x)g, %(y)g
                string color %(color)d
                string rot %(angle)f
                string font %(font)d
                string just %(just)d
                string char size %(size)f
                string def "%(text)s"
            """ % locals()
            self.write(strg)

    c=GracePlot.colors 
    stylecolors=[c.green,  c.blue, c.red, c.orange, c.magenta, c.black]
    s1, s2, s3, s4, s5, s6 =[
            GracePlot.Symbol(symbol=GracePlot.symbols.circle, fillcolor=sc, size=0.3, 
            linestyle=GracePlot.lines.none) for sc in stylecolors
        ]
    l1, l2, l3, l4, l5, l6=[
        GracePlot.Line(type=GracePlot.lines.solid, color=sc, linewidth=2.0) for sc in stylecolors]  
            
    noline=GracePlot.Line(type=GracePlot.lines.none)                

    graceSession=myGrace(width=11, height=8)
    
    g=graceSession[0]
    g.xlimit(-1,16)
    g.ylimit(-1,22)
    
    for row in range(16):
        for col in range(16):
            row*16+col
            graceSession.write_string(text=r"\x\#{%02x}"%(row*16+col), x=col, y=row, just=2, color=1, size=1.5)
        
    alphabet="".join(map(lambda x: "%("+x+")s", _greeklowernames)) +"%(aleph)s %(trademark-serif)s %(trademark-sans)s"
    print alphabet
    graceSession.write_string(text=format_python_to_grace(alphabet), x=0, y=17, just=0, color=1, size=1.5)
    
    alphabet="".join(map(lambda x: "%("+x+")s", _greekuppernames)) 
    print alphabet
    graceSession.write_string(text=format_python_to_grace(alphabet), x=0, y=18, just=0, color=1, size=1.5)
    alphabet=u"\N{GREEK CAPITAL LETTER PSI} \N{NOT SIGN} Goodbye \N{CIRCLED TIMES}"
    graceSession.write_string(text=translate_unicode_to_grace(alphabet), x=0, y=19, just=0, color=1, size=1.5)
    graceSession.redraw(True)
    
