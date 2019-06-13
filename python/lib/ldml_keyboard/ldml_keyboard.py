#!/usr/bin/python
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the University nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

from xml.etree import ElementTree as et
import itertools, re, os, sys
from collections import namedtuple
from copy import copy
import unicodedata as ud
from functools import reduce

try:
    from . import UnicodeSets, trie
except ValueError:
    if __name__ == '__main__':
        sys.path.insert(0, os.path.dirname(__file__))
        import UnicodeSets, trie

try: unicode
except NameError:
    unicode = str

# python3 etree has no way to get context info during parsing
# Use an external module if we want real error positions
try:
    from palaso.etree import XMLParser as EXMLParser
except ImportError:
    # Capture the start of each element for error reporting. py2 only. Harmless in py3.
    class EXMLParser(et.XMLParser):
        def __init__(self, **kw):
            super(EXMLParser, self).__init__(**kw)
            self.positions = {}
        def _start(self, tag, attrib_in):
            res = super(EXMLParser, self)._start(tag, attrib_in)
            self.positions[id(res)] = (self.parser.CurrentLineNumber, self.parser.CurrentColumnNumber)
            return res
        def _start_list(self, tag, attrib_in):
            res = super(EXMLParser, self)._start_list(tag, attrib_in)
            self.positions[id(res)] = (self.parser.CurrentLineNumber, self.parser.CurrentColumnNumber)
            return res

# A Syntax Error with positional context
class ESyntaxError(SyntaxError):
    def __init__(self, context, msg, parser, fname=None):
        lineno, offset = parser.positions.get(id(context), (0, 0))
        filename = fname
        super(ESyntaxError, self).__init__(msg, (filename, lineno, offset, msg))

CharCode = namedtuple('CharCode', ['primary', 'tertiary_base', 'tertiary', 'prebase'])
SortKey = namedtuple('SortKey', ['primary', 'index', 'tertiary', 'tiebreak'])
MatchResult = namedtuple('MatchResult', ['offset', 'length', 'rule', 'morep'])

class Keyboard(object):

    def __init__(self, path, imported=False):
        self.keyboards = []
        self.fallbacks = []
        self.modifiers = {}
        self.transforms = {}
        self.settings = {}
        self.history = []
        self.context = ""
        self.fname = path
        self.parse(path)
        self.tracing = False
        self.imported = imported

    def _addrules(self, element, transform, onlyifin=None, context=None):
        if transform not in self.transforms:
            self.transforms[transform] = Rules(transform)
        rules = self.transforms[transform]
        if context is not None:
            e = ESyntaxError(context, "", self.xmlParser, self.fname)
        else:
            e = SyntaxError("")
        for m in element:
            em = ESyntaxError(m, "", self.xmlParser, self.fname)
            rules.append(m, onlyifin, error=em)

    def parse(self, fname, imported=False):
        '''Read and parse an LDML keyboard layout file'''
        self.fname = fname
        xmlParser = EXMLParser()
        self.xmlParser = xmlParser
        doc = et.parse(fname, parser=self.xmlParser)
        for c in doc.getroot():
            if c.tag == 'keyMap':
                kind = None
                if 'modifiers' in c.attrib:
                    for m in self.parse_modifiers(c.get('modifiers')):
                        mod = " ".join(m)
                        testkind = self.modifiers.get(mod, None)
                        if kind == None:
                            kind = testkind
                        elif kind != testkind:
                            raise ESyntaxError(c, "Unmergeable keyMaps found for modifier: {}".format(m), \
                                              self.xmlParser, fname=self.fname)
                else:
                    kind = self.modifiers.get("", None)
                if kind is None:
                    kind = len(self.keyboards)
                    self.keyboards.append({})
                    if 'modifiers' in c.attrib:
                        for m in self.parse_modifiers(c.get('modifiers')):
                            self.modifiers[" ".join(m)] = kind
                    else:
                        self.modifiers[""] = kind
                maps = self.keyboards[kind]
                for m in c:
                    maps[m.get('iso')] = ud.normalize('NFD', UnicodeSets.struni(m.get('to')))
                self.fallbacks.append(c.get('fallback', "").split(' '))
            elif c.tag == 'transforms':
                t = c.get('type')
                self._addrules(c, t, context=c)
            elif c.tag == 'reorders':
                if self.keyboards:
                    testset = set(x for m in self.keyboards for v in m.values() for x in v)
                else:
                    testset = None
                self._addrules(c, 'reorder', onlyifin=testset, context=c)
            elif c.tag == 'backspaces':
                self._addrules(c, 'backspace', context=c)
            elif c.tag == 'settings' and not imported:
                self.settings.update(c.attrib)
            elif c.tag == 'import':
                for base in (os.path.dirname(fname),
                        os.path.join(os.path.dirname(__file__), '../../..')):
                    newfname = os.path.join(base, c.get('path'))
                    if os.path.exists(newfname):
                        self.parse(newfname, imported=True)
                        self.fname = fname
                        self.xmlParser = xmlParser
                        break
                else:
                    raise ImportError("Can't import {}".format(c.get('path')))

    def process_string(self, txt):
        '''Process a sequence of keystrokes expressed textually into a list
           of contexts giving the output after each keystroke'''
        self.initstring()
        keys = re.findall(r'\[\s*(.*?)\s*\]', txt)
        res = []
        for k in keys:
            words = k.split()
            modifiers = [x.lower() for x in words[:-1]]
            yield self.process(words[-1], modifiers)

    def parse_modifiers(self, modifiers):
        '''Flatten a modifier list into a list of possible modifiers'''
        resn = []
        reso = []
        for m in modifiers.lower().split():
            for mod in m.lower().split('+'):
                if mod.endswith("?"):
                    reso.append(mod)
                else:
                    resn.append(mod)
        yield sorted(resn)
        for i in range(len(reso)):
            for c in itertools.combinations(reso, i):
                yield sorted(resn + list(c))

    def initstring(self, ctext=""):
        '''Prepare to start processing a sequence of keystrokes'''
        self.history = []
        if ctext != "":
            self.history.append(Context(ctext, tracing=self.tracing))

    def undo(self, num=1):
        if num < len(self.history):
            self.history = self.history[:-num]
        else:
            self.history = []

    def process(self, k, mods, context=None):
        '''Process and record the results of a single keystroke given previous history'''
        chars = ud.normalize('NFD', self.map_key(k, mods))
        if isinstance(context, Context):
            ctxt = context
        elif context is not None:
            ctxt = self._ContextFromString(ud.normalize('NFD', context+chars))
        elif not len(self.history):
            ctxt = Context(chars, tracing=self.tracing)
        else:
            ctxt = self.history[-1].clone(chars, tracing=self.tracing)
        
        if k == 'BKSP':
            # normally we would simply undo, but test the backspace transforms
            if not self.process_backspace(ctxt, 'backspace'):
                return self.error(ctxt)
        else:
            if not self.process_simple(ctxt):
                self.error(ctxt)
            self.process_reorder(ctxt)
            if not self.process_simple(ctxt, 'final', handleSettings=False):
                self.error(ctxt)
        if context is None:
            self.history.append(ctxt)
        return ctxt

    def error(self, ctxt = None):
        '''Set error state'''
        if ctxt is not None:
            pass
        elif not len(self.history):
            ctxt = Context(tracing=self.tracing)
        else:
            ctxt = self.history[-1].clone()
        ctxt.error = 1
        return ctxt

    def map_key(self, k, mods):
        '''Apply the appropriate keyMap to a keystroke to get some chars'''
        modstr = " ".join(sorted(x.lower() for x in mods))
        if modstr not in self.modifiers:
            return ""
        kind = self.modifiers[modstr]
        if k in self.keyboards[kind]:
            return UnicodeSets.struni(self.keyboards[kind][k])
        for f in self.fallbacks[kind]:
            if f in self.modifiers:
                find = self.modifiers[f]
                if k in self.keyboards[find]:
                    return UnicodeSets.struni(self.keyboards[find][k])
        return ""

    def _ContextFromString(self, s):
        '''Create a new processing context from a string '''
        res = Context(tracing=self.tracing)
        (before, changed, _) = self._unreorder(s)
        olds = before + changed
        for t in ('reorder', 'final'):
            res.set_output(t, olds)
            res.set_offset(t, len(before))
            res.set_partial(t, len(olds) - len(before))
        for t in ('base', 'simple'):
            res.set_output(t, olds)
            res.set_offset(t, len(olds))
        res.trace("Initialise with string " + repr(s))
        return res

    def process_empty(self, context, ruleset):
        '''Copy layer input to output'''
        context.reset_output(ruleset)
        output = context.input(ruleset)[context.offset(ruleset):]
        context.results(ruleset, context.offset(ruleset), len(output), output)

    def process_simple(self, context, ruleset='simple', handleSettings=True):
        '''Handle a simple replacement transforms type'''
        if ruleset not in self.transforms:
            self.process_empty(context, ruleset)
            return True
        trans = self.transforms[ruleset]
        if handleSettings:
            partial = self.settings.get('transformPartial', "") == "hide"   # partial match
            fail = self.settings.get('transformFailure', "") == 'omit'      # unmatched sequence
            fallback = self.settings.get('fallback', "") == 'omit'          # unmatched first char
        else:
            partial = False
            fail = False
            fallback = False

        context.reset_output(ruleset)
        curr = context.offset(ruleset)
        instr = context.input(ruleset)
        ind = context.index(ruleset)
        while curr < len(instr):
            r = trans.match(instr, curr)
            if r.rule is not None:
                if r.offset > 0:    # there is a @before context, output it
                    context.results(ruleset, curr, r.offset, instr[curr:curr+r.offset], rule=r.rule)
                    r.length -= r.offset
                    curr += r.offset
                if getattr(r.rule, 'error', 0):
                    context.error = True
                newout = ud.normalize('NFD', UnicodeSets.struni(getattr(r.rule, 'to', "")))
                # output replaced string
                context.results(ruleset, curr, r.length, newout, rule=r.rule)
                if ruleset == 'final':
                    curr += r.length
                    continue    # no reprocessing of final type transforms
                # allow reprocessing of input transforms only. This gives us rotors, for example.
                for i in range(len(newout), 0, -1):
                    # do a test match on the suffix
                    t = trans.match(context.output(ruleset), len(context.output(ruleset)) - i)
                    # if the suffix is an incomplete match then remove that much from the output
                    # and put back onto input.
                    if t.morep:
                        instr = context.put_back(ruleset, instr, r.length, curr, i)
                        curr = 0
                        break
                else:   # if nothing put back, simply advance over what we have used
                    curr += r.length
            elif r.length == 0 and not r.morep and not fallback:     # abject failure
                context.results(ruleset, curr, 1, instr[curr:curr+1], comment="Fallthrough")
                curr += 1
            elif r.morep and not partial:       # if incomplete match then store partial results
                l = r.length or 1
                context.partial_results(ruleset, l, instr[curr:curr+l], comment="Partial")
                curr += l
            elif r.length != 0 and not r.morep and not fail:        # matched but no rule, just copy
                context.results(ruleset, curr, r.length, instr[curr:curr+r.length], comment="Fail")
                curr += r.length
            else:               # partial match waiting for more input
                break
        return True

    def _sort(self, begin, end, chars, keys, rev=False):
        '''Sort a list of chars based on a corresponding list of keys. If no base (key.primary==0)
           in the sequence, insert a U+200B as a base character. If rev remove any U+200B
           immediately preceding a prevowel (after sorting). Returns (prefix, sorted, suffix)'''
        s = chars[begin:end]
        k = keys[:end-begin]
        if not len(s):
            return (chars[:begin], u"", chars[end:])
        # if there is no base, insert one
        if not rev and (0, 0) not in [(x.primary, x.tertiary) for x in k]:
            s += u"\u200B"
            k.append(SortKey(0, 0, 0, 0))  # push this to the front
        # sort key is (primary, tertiary_base, tertiary, string index as tie break)
        res = u"".join(s[y] for y in sorted(range(len(s)), key=lambda x:k[x]))
        if rev:
            # remove a \u200B if the cluster start, after a prevowel
            foundpre = False
            for i, key in enumerate(sorted(k)):
                if key.primary < 0:
                    foundpre = True
                elif key.primary > 0:
                    break
                elif foundpre and res[i] == u"\u200B":
                    res = res[:i] + res[i+1:]
                    break
        return (chars[:begin], res, chars[end:])

    def _padlist(self, val, num):
        '''Fill out short value lists to full length of matched string '''
        boolmap = {'false' : 0, 'true': 1}
        res = [boolmap.get(x.lower(), x) for x in val.split()]
        if len(res) < num:
            res += [res[-1]] * (num - len(res))
        return res

    def _get_charcodes(self, instr, curr, trans, rev=False):
        '''Returns a list of some CharCodes, 1 per char, for the string at curr''' 
        r = trans.revmatch(instr, curr) if rev else trans.match(instr, curr)
        if r.rule is not None:
            if r.offset > 0:
                return [CharCode(0, 0, 0, False)] * r.offset
            orders = [int(x) for x in self._padlist(getattr(r.rule, 'order', "0"), r.length)]
            bases = [bool(x) for x in self._padlist(getattr(r.rule, 'tertiary_base', "false"), r.length)]
            tertiaries = [int(x) for x in self._padlist(getattr(r.rule, 'tertiary', "0"), r.length)]
            prebases = [int(x) for x in self._padlist(getattr(r.rule, 'prebase', "0"), r.length)]
            return [CharCode(orders[i], bases[i], tertiaries[i], prebases[i]) for i in range(r.length)]
        else:
            return [CharCode(0, 0, 0, False)]

    def process_reorder(self, context, ruleset='reorder'):
        '''Handle the reorder transforms'''
        if ruleset not in self.transforms:
            self.process_empty(context, ruleset)
            return
        trans = self.transforms[ruleset]
        instr = context.input(ruleset)
        context.reset_output(ruleset)

        # scan for start of sortable run. Normally empty
        startrun = context.offset(ruleset)
        curr = startrun
        codes = self._get_charcodes(instr, curr, trans)
        # if not at start of a run
        if curr and (codes[0].primary > 0 and not codes[0].prebase or codes[0].tertiary):
            # walk backwards looking for the start of a run. May occur if the addition of a character
            # changes the category of a preceding character from cluster start to cluster medial.
            curr -=1
            while curr > 0:
                # scan back for charcodes, since whether something is primary may be contextual
                codes = self._get_charcodes(instr, len(instr) - 1 - curr, trans, rev=True)
                curr -= len(codes)
                if not codes[0].tertiary and (codes[0].primary == 0 or codes[0].prebase):
                    # found the start of a run, perhaps. Assumes primary is always at start of sequence.
                    break
            while curr > 0:
                # now scan back to find the end of the previous cluster (e.g. there are prevowels)
                codes = self._get_charcodes(instr, len(instr) - 1 - curr, trans, rev=True)
                if codes[-1].tertiary or (codes[-1].primary and not codes[-1].prebase):
                    # found it
                    break
                curr -= len(codes)
            # rewind everything accordingly
            context.backup_to(ruleset, curr)

        # now scan for start of cluster, e.g. if string doesn't start with cluster start
        while curr < len(instr):
            codes = self._get_charcodes(instr, curr, trans)
            for c in codes:
                if not c.tertiary and (c.prebase or c.primary == 0):
                    break
                curr += 1
            else:
                continue        # if didn't break inner loop, don't break outer loop
            break               # if we broke in the inner loop, break the outer loop
        if curr > startrun:     # just copy the odd characters across
            context.replace_middle(ruleset, startrun, curr - startrun, instr[startrun:curr])

        # iterate over sorted clusters
        for pre, ordered, post in self._reorder(instr, startrun=curr, end=context.len(ruleset), \
                                                ruleset=ruleset, ctxt=context):
            if len(post):
                # append and update processed marker
                context.results(ruleset, curr, len(ordered), ordered, comment="Reordered")
            else:
                # just append, ready to reprocess again
                context.partial_results(ruleset, len(ordered), ordered, comment="Partial reorder")

    def _reorder(self, instr, startrun=0, end=0, ruleset='reorder', ctxt=None):
        '''Presumes we are at the start of a cluster. Analyses a single cluster and orders it.'''
        trans = self.transforms[ruleset]
        curr = startrun
        keys = [None] * (len(instr) - startrun)
        isinit = True           # inside the start of a run (.{prebase}* .{order==0 && tertiary==0})
        currprimary = 0
        currbaseindex = curr
        while curr < end:
            codes = self._get_charcodes(instr, curr, trans)
            if ctxt is not None:
                ctxt.trace('Reorder codes({}) for "{}" {}'.format(len(codes), 
                                    instr[curr:curr+len(codes)].encode("unicode_escape"), codes))
            for i, c in enumerate(codes):               # calculate sort key for each character in turn
                if c.tertiary and curr + i > startrun:  # can't start with tertiary, treat as primary 0
                    # use remembered values for primary and positions
                    key = SortKey(currprimary, currbaseindex, c.tertiary, curr + i)
                else:
                    key = SortKey(c.primary, curr + i, 0, curr + i)
                    if c.primary == 0 or c.tertiary_base:   # primary 0 is always a tertiary base
                        currprimary = c.primary             # track current primary for tertiaries
                        currbaseindex = curr + i            # track position of primary

                # are we past initials in a cluster (prebase and primaries)
                if ((key.primary != 0 or key.tertiary != 0) and not c.prebase) \
                        or (c.prebase and curr + i > startrun \
                            and keys[curr+i-startrun-1].primary == 0):  # prebase can't have tertiary
                    isinit = False  # After the prefix, so any following prefix char starts a new run

                # identify a run boundary
                if not isinit and ((key.primary == 0 and key.tertiary == 0) or c.prebase):
                    # output sorted run and reset for new run
                    yield self._sort(0, curr + i - startrun, instr[startrun:], keys)
                    startrun = curr + i
                    keys = [None] * (len(instr) - startrun)
                    isinit = True
                keys[curr+i-startrun] = key
            curr += len(codes)
        # yield a final result with the residue and no post text
        if curr > startrun:
            yield self._sort(0, curr-startrun, instr[startrun:], keys)
        else:
            yield ("", "", "")      # indicates the end of a cluster

    def _unreorder(self, instr):
        '''Create a string that when reordered gives the input string.
           This relies on well designed reorders, but is generally what happens.
           Returns (unchanged prefix string, unreordered single cluster, "")'''
        end = 0
        try:
            trans = self.transforms['reorder']
        except KeyError:
            return (instr, "", "")
        hitbase = False
        keys = []
        tertiaries = []
        while end < len(instr):
            # work backwards from the end of the string
            codes = self._get_charcodes(instr, end, trans, rev=True)
            for c in codes:
                # having hit a cluster prefix do we now hit the end of the last cluster?
                if hitbase and (c.primary >= 0 or c.tertiary > 0):
                    break
                # hit the end of a cluster prefix
                elif c.primary == 0 and c.tertiary == 0:
                    hitbase = True
                    keys.append(SortKey(0, -end, 0, -end))
                    # bases are always tertiary_bases so update tertiary references
                    for e in tertiaries:
                        keys[e] = SortKey(0, -end, keys[e].tertiary, keys[e].tiebreak)
                    tertiaries = []
                # tertiary characters get given a key and a reference to update
                elif c.tertiary != 0:
                    tertiaries.append(end)
                    keys.append(SortKey(0, -end, c.tertiary, -end))
                # normal case
                else:
                    # if this is reordered before the start of a cluster, reorder it to the end
                    # doesn't really matter where it ends up, it'll get sorted back
                    v = c.primary + (127 if c.primary < 0 else 0)
                    # sort prebases before base
                    if c.prebase:
                        v = c.prebase - 10
                    keys.append(SortKey(v, -end, c.tertiary, -end))
                    # update tertiary references
                    if c.tertiary_base:
                        for e in tertiaries:
                            keys[e] = SortKey(v, -end, keys[e].tertiary, keys[e].tiebreak)
                        tertiaries = []
                end += 1
            else:
                continue
            break
        keys = list(reversed(keys))
        res = self._sort(len(instr) - len(keys), len(instr), instr, keys, rev=True)
        return res

    def _default_backspace(self, instr):
        """Consume one character from the pre-reordered text.
           Returns replacement output string, length of output consumed,
           replacement pre reordered text, length of pre reordered text consumed."""
        #instr = ud.normalize('NFD', instr)
        if 'reorder' not in self.transforms:
            shorter = ud.normalize('NFD', ud.normalize('NFC', instr)[:-1])
            difflen = len(instr) - len(shorter)
            return ("", difflen, "", difflen)
        # derive a possible input string to reorder
        (orig, simple, _) = self._unreorder(instr)
        olen = len(orig)
        slen = len(simple)
        # delete one char from it. Treat as NFC.
        simple = ud.normalize('NFD', ud.normalize('NFC', simple)[:-1])
        nslen = len(simple)
        # recalculate output as a result
        (pref, res, post) = list(self._reorder(simple, end=len(simple), ruleset='reorder'))[0]
        # Find the point at which a change occurred in unreordered text
        for i in range(nslen):
            if simple[i] != instr[olen + i]:
                slen += i
                simple = simple[i:]
                break
        else:
            slen -= nslen
            simple = u""
        # Find the point at which the output differs from input
        length = len(res)
        for i in range(len(res)):
            if res[i] != instr[olen + i]:
                length = i
                res = res[i:]
                break
        else:
            res = u""
        # return the trimmed down reordered string, the length,
        # the trimmed down string that needs to be reordered and length
        return (res, len(instr) - olen - length, simple, slen)

    def process_backspace(self, context, ruleset='backspace'):
        '''Handle the backspace transforms in response to bksp key'''
        instr = context.output('final')
        instrlen = len(instr)
        rule = None
        if ruleset not in self.transforms:
            (res, length, simple, slen) = self._default_backspace(instr)
        else:
            trans = self.transforms[ruleset]
            # find and process one rule
            r = trans.revmatch(instr)
            if r.rule is not None:
                if getattr(r.rule, 'error', 0): return False
                length = r.length
                res = ud.normalize('NFD', UnicodeSets.struni(getattr(r.rule, 'to', "")))
                # derive possible input string to lead to desired result
                (orig, simple, _) = self._unreorder(instr[:-length]+res)
                rule = r.rule
                # this doesn't have to be accurate since we reset offsets
                slen = instrlen - len(orig)
            else:       # no rule, so just remove a single character
                (res, length, simple, slen) = self._default_backspace(instr)
        # replace the various between pass strings
        for x in ('base', 'simple'):
            context.replace_end(x, slen, simple, rule=rule)
        for x in ('reorder', 'final'):
            context.replace_end(x, length, res, rule=rule)
        return True


class Rules(object):
    '''Corresponds to a transforms element in an LDML file'''
    def __init__(self, ruletype):
        self.type = ruletype
        self.trie = trie.Trie()

    def append(self, transform, onlyifin=None, error=None):
        '''Insert or transform element into this set of rules'''
        f = unicode(transform.get('from'))
        if f is None and error is not None:
            error.msg = "Missing @from attribute in rule"
            raise error

        r = Rule(transform)
        if error is not None:
            r.context = (error.filename, error.lineno, error.offset)
        self.trie.append(f, transform.get('before', ''), \
                transform.get('after', ''), r, filterlist=onlyifin, normal="NFD")

    def match(self, s, ind=0):
        '''Finds the merged rule for the given passed in string.
           Returns (offset, length, rule, morep) as a MatchRule where length
           is how far to advance the cursor. Offset is how far to skip before
           replacing. Rule is the Rule object and morep is a boolean that 
           says whether if given more characters in the string, further
           matching may have occurred (see settings/@transformPartial.'''
        return MatchResult(*self.trie.match(s, ind))

    def revmatch(self, s, ind=0):
        '''Searches backwards from the end-ind for a match'''
        return MatchResult(*self.trie.revmatch(s, ind))

class Rule(object):
    '''A trie element that might do something. Corresponds to a
       flattened transform in LDML'''
    def __init__(self, transform):
        for k, v in transform.items():
            if k in ('from', 'before', 'after'):
                continue
            setattr(self, k, v)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return "Rule(" + ", ".join(sorted("{}={}".format(k, getattr(self, k)) \
                                    for k in dir(self) if not k.startswith('_'))) + ")"

    def _newmerge(self, other):
        res = Rule({})
        for k, v in ((k,getattr(self, k)) for k in dir(self) if not k.startswith('_')):
            setattr(res, k, v)
        for k, v in ((k,getattr(other, k)) for k in dir(other) if not k.startswith('_')):
            setattr(res, k, v)
        return res

class Context(object):
    '''Holds the processed state of each layer after a keystroke. Keeps track of
       the currently processed strings after each pass.'''
    slotnames = {
        'base' : 0,
        'simple' : 1,
        'reorder' : 2,
        'final' : 3
    }
    def __init__(self, chars="", tracing=False):
        self.outputs = [""] * len(self.slotnames)   # stuff to pass to next layer
        self.outputs[0] = chars                     # and copy it to its output
        self.offsets = [0] * len(self.slotnames)    # pointer into last layer output
                                                    # corresponding to end of stables
        self.partials = [0] * len(self.slotnames)   # how much of my output is partial
        self.offsets[0] = len(chars)
        self.error = 0                              # are we in an error state?
        self.enable_tracing = tracing
        self.tracing_events = []
        self.trace("Initialise with {}".format(repr(chars)))

    def clone(self, chars="", tracing=False):
        '''Copy a context and add some more input to the result'''
        res = Context("")
        res.outputs = self.outputs[:]
        res.offsets = self.offsets[:]
        res.partials = self.partials[:]
        res.outputs[0] += chars
        res.offsets[0] += len(chars)
        res.enable_tracing = self.enable_tracing
        res.tracing_events = []
        res.tracing = tracing
        res.trace("Cloning as {}".format(repr(res.outputs[0])))
        return res

    def __str__(self):
        return self.outputs[-1]

    def index(self, name='base'):
        return self.slotnames[name]

    def len(self, name='simple'):
        '''Return length of input to this layer'''
        return len(self.outputs[self.slotnames[name]-1])

    def input(self, name='simple'):
        '''Return full input string to this layer'''
        return self.outputs[self.slotnames[name]-1]

    def output(self, name='simple'):
        return self.outputs[self.index(name)]

    def offset(self, name='simple'):
        '''Returns the offset into input string that isn't in stables'''
        return self.offsets[self.slotnames[name]]

    def set_output(self, name, s):
        self.outputs[self.index(name)] = s

    def set_offset(self, name, i):
        self.offsets[self.index(name)] = i

    def set_partial(self, name, i):
        self.partials[self.index(name)] = i

    def reset_output(self, name):
        '''Prepare output based on stables ready for more output to be added'''
        ind = self.index(name)
        if self.partials[ind]:
            self.outputs[ind] = self.outputs[ind][:-self.partials[ind]]
            self.partials[ind] = 0
        self.trace("Resetting {} by {} to {}, probably on transform entry".format(name, \
                    self.partials[ind], unicode(self.outputs[ind]).encode('unicode_escape')))

    def backup_to(self, name, index):
        '''Resets input and all subsequent outputs to given index'''
        ind = self.index(name)
        diff = self.offsets[ind] - index
        for i in range(ind, len(self.outputs)):
            self.offsets[i] -= diff
            self.outputs[i] = self.outputs[i][:-diff]

    def partial_results(self, name, length, res, rule=None, comment=None):
        ind = self.index(name)
        self.outputs[ind] += res
        self.partials[ind] += length
        self.trace_replacement("Add Partial", length, name, res, rule=rule, comment=comment)

    def results(self, name, start, length, res, rule=None, comment=None):
        '''Remove from input, in effect, and update offsets'''
        ind = self.index(name)
        leftlen = len(self.outputs[ind-1]) - start - length
        self.outputs[ind] += res
        if leftlen >= self.partials[ind-1]:
            self.offsets[ind] = start + length
            self.partials[ind] = 0
        else:
            self.partials[ind] += len(res)
        self.trace_replacement("Consumed", length, name, res, rule=rule, comment=comment)

    def replace_end(self, name, backup, res, rule=None, comment=None):
        '''Backup specified number of characters and replace with given string'''
        ind = self.index(name)
        out = self.outputs[ind]
        self.outputs[ind] = out[:-backup] + res
        if ind < len(self.outputs) - 1:
            if len(self.outputs[ind]) < self.offsets[ind+1]:
                self.offsets[ind+1] = len(self.outputs[ind])
        if self.partials[ind] >= backup - len(res):
            self.partials[ind] += len(res) - backup
        elif self.partials[ind] != 0:
            self.partials[ind] = len(res)
        if ind == 0:
            self.offsets[ind] = len(self.outputs[ind])
        self.trace_replacement("Truncated", backup, name, res, rule=rule, comment=comment)

    def replace_middle(self, name, start, length, res, rule=None, comment=None):
        '''Replace in the middle of a string'''
        ind = self.index(name)
        out = self.outputs[ind]
        self.outputs[ind] = out[:start] + res + out[start+length:]
        if start < self.offsets[ind]:
            self.offsets[ind] = start
        if ind > 0:
            self.partials[ind] = max(self.partials[ind], self.partials[ind-1])
        if self.partials[ind] > len(out) - start - length:
            self.partials[ind] += len(res) - length
        self.trace_replacement("Middle", length, name, res, rule=rule, comment=comment)

    def put_back(self, name, instr, inlength, inoffset, outlength):
        '''Remove outlength chars from output and return them as a prefix to replace
           inlength chars from the instr at the given inoffset. So the return is:
           some_output + instr[inoffset+inlength:]'''
        ind = self.index(name)
        self.offsets[ind] -= inlength
        instr = self.outputs[ind][-outlength:] + instr[inoffset + inlength:]
        self.outputs[ind] = self.outputs[ind][:-outlength]
        return instr

    def trace_replacement(self, txt, length, name, res, rule=None, comment=None):
        if self.enable_tracing:
            extra = ""
            if comment is not None:
                extra += " " + comment
            if rule is not None and hasattr(rule, 'context'):
                extra += " in rule {}:{} col {}".format(*rule.context)
            self.trace("{}({}) from {}, replaced with {}{}".format(txt, length, name, res.encode("unicode_escape"), extra))

    def trace(self, txt):
        if self.enable_tracing:
            self.tracing_events.append(txt)

def main():
    '''Process a testfile of key sequences, one sequence per line,
       to give test results: comma separated for each keystroke,
       one sequence per line'''
    import argparse, codecs, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('file',help='Input LDML keyboard file')
    parser.add_argument('-t','--testfile',help='File of key sequences, one per line')
    parser.add_argument('-o','--outfile',help='Where to send results')
    args = parser.parse_args()

    kbd = Keyboard(args.file)
    if args.outfile:
        outfile = codecs.open(args.outfile, "w", encoding="utf-8")
    else:
        outfile = codecs.EncodedFile(sys.stdout, "utf-8")
    with open(args.testfile) as inf:
        for l in inf.readlines():
            res = list(kbd.process_string(l))
            outfile.write(u", ".join(map(unicode, res)) + u"\n")
    outfile.close()

if __name__ == '__main__':
    main()
