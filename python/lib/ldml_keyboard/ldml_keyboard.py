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

try:
    from . import UnicodeSets, trie
except ValueError:
    if __name__ == '__main__':
        sys.path.insert(0, os.path.dirname(__file__))
        import UnicodeSets, trie

# Capture the start of each element for error reporting
class EXMLParser(et.XMLParser):
    def _start(self, tag, attrib_in):
        res = super(EXMLParser, self)._start(tag, attrib_in)
        res.error_pos = (self.parser.CurrentLineNumber, self.parser.CurrentColumnNumber)
        return res
    def _start_list(self, tag, attrib_in):
        res = super(EXMLParser, self)._start_list(tag, attrib_in)
        res.error_pos = (self.parser.CurrentLineNumber, self.parser.CurrentColumnNumber)
        return res

class ESyntaxError(SyntaxError):
    def __init__(self, context, msg, fname=None):
        self.lineno, self.offset = context.error_pos
        if fname is not None:
            self.filename = fname

CharCode = namedtuple('CharCode', ['primary', 'tertiary_base', 'tertiary', 'prebase'])
SortKey = namedtuple('SortKey', ['primary', 'index', 'tertiary', 'tiebreak'])
MatchResult = namedtuple('MatchResult', ['offset', 'length', 'rule', 'morep'])

class Keyboard(object):

    def __init__(self, path):
        self.keyboards = []
        self.fallbacks = []
        self.modifiers = {}
        self.transforms = {}
        self.settings = {}
        self.history = []
        self.context = ""
        self.fname = path
        self.parse(path)

    def _addrules(self, element, transform, onlyifin=None, context=None):
        if transform not in self.transforms:
            self.transforms[transform] = Rules(transform)
        rules = self.transforms[transform]
        if context is not None:
            e = ESyntaxError(context, "", self.fname)
        else:
            e = SyntaxError("")
        for m in element:
            rules.append(m, onlyifin, error=e)

    def parse(self, fname):
        '''Read and parse an LDML keyboard layout file'''
        self.fname = fname
        doc = et.parse(fname, parser=EXMLParser())
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
                            raise ESyntaxError(c, "Unmergeable keyMaps found for modifier: {}".format(m), fname=self.fname)
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
                    maps[m.get('iso')] = UnicodeSets.struni(m.get('to'))
                self.fallbacks.append(c.get('fallback', "").split(' '))
            elif c.tag == 'transforms':
                self._addrules(c, c.get('type'), context=c)
            elif c.tag == 'reorders':
                testset = set(x for m in self.keyboards for v in m.values() for x in v)
                self._addrules(c, 'reorder', onlyifin=testset, context=c)
            elif c.tag == 'backspaces':
                self._addrules(c, 'backspace', context=c)
            elif c.tag == 'settings':
                self.settings.update(c.attrib)
            elif c.tag == 'import':
                for base in (os.path.dirname(fname),
                        os.path.join(os.path.dirname(__file__), '../../..')):
                    newfname = os.path.join(base, c.get('path'))
                    if os.path.exists(newfname):
                        self.parse(newfname)
                        break
                else:
                    raise ImportError("Can't import {}".format(c.get('path')))

    def process_string(self, txt):
        '''Process a sequence of keystrokes expressed textually into a list
            of contexts giving the output after each keystroke'''
        self.initstring()
        keys = re.findall(ur'\[\s*(.*?)\s*\]', txt)
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

    def initstring(self):
        '''Prepare to start processing a sequence of keystrokes'''
        self.history = []

    def process(self, k, mods):
        '''Process and record the results of a single keystroke given previous history'''
        chars = self.map_key(k, mods)
        if not len(self.history):
            ctxt = Context(chars)
        else:
            ctxt = self.history[-1].clone(chars)
        
        if k == 'BKSP':
            # normally we would simply undo, but test the backspace transforms
            if not self._process_backspace(ctxt, 'backspace'):
                return self.error(ctxt)
        else:
            if not self._process_simple(ctxt):
                return self.error()
            self._process_reorder(ctxt)
            if not self._process_simple(ctxt, 'final', handleSettings=False):
                return self.error()
        self.history.append(ctxt)
        return ctxt

    def error(self):
        '''Set error state'''
        if not len(self.history):
            res = Context()
        else:
            res = self.history[-1].clone()
        res.error = 1
        return res

    def map_key(self, k, mods):
        '''Apply the appropriate keyMap to a keystroke to get some chars'''
        modstr = " ".join(sorted(mods))
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

    def _process_empty(self, context, ruleset):
        '''Copy layer input to output'''
        context.reset_output(ruleset)
        output = context.input(ruleset)[context.offset(ruleset):]
        context.results(ruleset, len(output), output)

    def _process_simple(self, context, ruleset='simple', handleSettings=True):
        '''Handle a simple replacement transforms type'''
        if ruleset not in self.transforms:
            self._process_empty(context, ruleset)
            return True
        trans = self.transforms[ruleset]
        if handleSettings:
            partial = self.settings.get('transformPartial', "") == "hide"
            fail = self.settings.get('transformFailure', "") == 'omit'
            fallback = self.settings.get('fallback', "") == 'omit'
        else:
            partial = False
            fail = False
            fallback = False

        context.reset_output(ruleset)
        curr = context.offset(ruleset)
        instr = context.input(ruleset)
        while curr < len(instr):
            r = trans.match(instr, curr)
            if r.rule is not None:
                if getattr(r.rule, 'error', 0): return False
                if r.offset > 0:
                    context.results(ruleset, r.offset, instr[curr:curr+r.offset])
                    r.length -= r.offset
                    curr += r.offset
                context.results(ruleset, r.length, UnicodeSets.struni(getattr(r.rule, 'to', "")))
                curr += r.length
            elif r.length == 0 and not fallback and (not r.morep or not partial):     # abject failure
                context.results(ruleset, 1, instr[curr:curr+1])
                curr += 1
            else:               # partial match waiting for more input
                break
        return True

    def _sort(self, begin, end, chars, keys):
        s = chars[begin:end]
        k = keys[:end-begin]
        # if there is no base, insert one
        if (0, 0) not in [(x.primary, x.tertiary) for x in k]:
            s += u"\u200B"
            k += SortKey(0, 0, 0, 0)  # push this to the front
        # sort key is (primary, secondary, string index)
        return u"".join(s[y] for y in sorted(range(len(s)), key=lambda x:k[x]))

    def _padlist(self, val, num):
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
            prebases = [bool(x) for x in self._padlist(getattr(r.rule, 'prebase', "false"), r.length)]
            return [CharCode(orders[i], bases[i], tertiaries[i], prebases[i]) for i in range(r.length)]
        else:
            return [CharCode(0, 0, 0, False)]

    def _process_reorder(self, context, ruleset='reorder'):
        '''Handle the reorder transforms'''
        if ruleset not in self.transforms:
            self._process_empty(context, ruleset)
            return
        trans = self.transforms[ruleset]
        instr = context.input(ruleset)
        context.reset_output(ruleset)

        # scan for start of sortable run. Normally empty
        startrun = context.offset(ruleset)
        curr = startrun
        while curr < len(instr):
            codes = self._get_charcodes(instr, curr, trans)
            for c in codes:
                if c.prebase or c.primary == 0:
                    break
                curr += 1
            else:
                continue        # if didn't break inner loop, don't break outer loop
            break               # if we broke in the inner loop, break the outer loop
        if curr > startrun:     # just copy the odd characters across
            context.results(ruleset, curr - startrun, instr[startrun:curr])

        startrun = curr
        keys = [None] * (len(instr) - startrun)
        isinit = True           # inside the start of a run (.{prebase}* .{order==0 && tertiary==0})
        currprimary = 0
        currbaseindex = curr
        while curr < context.len(ruleset):
            codes = self._get_charcodes(instr, curr, trans)
            for i, c in enumerate(codes):               # calculate sort key for each character in turn
                if c.tertiary and curr + i > startrun:      # can't start with tertiary, treat as primary 0
                    key = SortKey(currprimary, currbaseindex, c.tertiary, curr + i)
                else:
                    key = SortKey(c.primary, curr + i, 0, curr + i)
                    if c.primary == 0 or c.tertiary_base:   # primary 0 is always a tertiary base
                        currprimary = c.primary
                        currbaseindex = curr + i

                if ((key.primary != 0 or key.tertiary != 0) and not c.prebase) \
                        or (c.prebase and curr + i > startrun \
                            and keys[curr+i-startrun-1].primary == 0):  # prebase can't have tertiary
                    isinit = False      # After the prefix, so any following prefix char starts a new run

                # identify a run boundary
                if not isinit and ((key.primary == 0 and key.tertiary == 0) or c.prebase):
                    # output sorted run and reset for new run
                    context.results(ruleset, curr + i - startrun,
                                    self._sort(startrun, curr + i, instr, keys))
                    startrun = curr + i
                    keys = [None] * (len(instr) - startrun)
                    isinit = True
                keys[curr+i-startrun] = key
            curr += len(codes)
        if curr > startrun:
            # output but don't store any residue. Reprocess it next time.
            context.outputs[context.index(ruleset)] += self._sort(startrun, curr, instr, keys)

    def _unreorder(self, instr):
        end = 0
        trans = self.transforms['reorder']
        hitbase = False
        keys = []
        tertiaries = []
        while end < len(instr):
            codes = self._get_charcodes(instr, end, trans, rev=True)
            for c in codes:
                if c.primary == 0 and c.tertiary == 0:
                    hitbase = True
                    keys.append(SortKey(0, -end, 0, -end))
                    for e in tertiaries:
                        keys[e] = SortKey(keys[e].primary, -end, keys[e].tertiary, keys[e].tiebreak)
                    tertiaries = []
                elif hitbase and not c.prebase and c.primary >= 0 and c.tertiary == 0:
                    break
                elif c.tertiary != 0:
                    tertiaries.append(end)
                    keys.append(SortKey(0, -end, c.tertiary, -end))
                else:
                    v = c.primary + (127 if c.primary < 0 else 0)
                    if c.prebase:
                        v = -c.prebase
                    keys.append(SortKey(c.primary, -end, c.tertiary, -end))
                    if c.tertiary_base:
                        for e in tertiaries:
                            keys[e] = SortKey(keys[e].primary, -end, keys[e].tertiary, keys[e].tiebreak)
                        tertiaries = []
                end += 1
            else:
                continue
            break
        keys = list(reversed(keys))
        res = self._sort(len(instr) - len(keys), len(instr), instr, keys)
        return res

    def _process_backspace(self, context, ruleset='backspace'):
        '''Handle the backspace transforms in response to bksp key'''
        if ruleset not in self.transforms:
            self.chomp(context)
        trans = self.transforms[ruleset]
        # reverse the string
        instr = context.outputs[-1][::-1]
        origlen = len(instr)
        # find and process one rule
        r = trans.revmatch(instr)
        if r.rule is not None:
            if getattr(r.rule, 'error', 0): return False
            instr[:rlength] = getattr(r.rule, 'to', "")
        else:       # no rule, so just remove a single character
            instr = instr[1:]
        # and reverse back again
        instr = instr[::-1]
        unorderedstr = self._unreorder(instr)
        for x in ('base', 'simple'):
            context.replace_end(x, len(unorderedstr) + origlen - len(instr), unorderedstr)
        for x in ('reorder', 'final'):
            context.replace_end(x, origlen, instr)
        return True


class Rules(object):
    '''Corresponds to a transforms element in an LDML file'''
    def __init__(self, ruletype):
        self.type = ruletype
        self.trie = trie.Trie()

    def append(self, transform, onlyifin=None, error=None):
        '''Insert or transform element into this set of rules'''
        f = transform.get('from')
        if f is None and error is not None:
            error.msg = "Missing @from attribute in rule"
            raise error

        self.trie.append(f, transform.get('before', ''), \
                transform.get('after', ''), Rule(transform), filterlist=onlyifin)

    def match(self, s, ind=0):
        '''Finds the merged rule for the given passed in string.
            Returns (offset, length, rule, morep) as a MatchRule where length is how far to advance the cursor.
            Offset is how far to skip before replacing. Rule is the Rule object and morep
            is a boolean that says whether if given more characters in the string, further
            matching may have occurred (see settings/@transformPartial.
        '''
        return MatchResult(*self.trie.match(s, ind))

    def revmatch(self, s, ind=0):
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
    '''Holds the processed state of each layer after a keystroke'''

    slotnames = {
        'base' : 0,
        'simple' : 1,
        'reorder' : 2,
        'final' : 3
    }
    def __init__(self, chars=""):
        self.stables = [""] * len(self.slotnames)   # stuff we don't need to reprocess
        self.outputs = [""] * len(self.slotnames)   # stuff to pass to next layer
        self.stables[0] = chars                     # basic input is always stable
        self.outputs[0] = chars                     # and copy it to its output
        self.offsets = [0] * len(self.slotnames)    # pointer into last layer output
                                                    # corresponding to end of stables
        self.offsets[0] = len(chars)
        self.error = 0                              # are we in an error state?

    def clone(self, chars=""):
        '''Copy a context and add some more input to the result'''
        res = Context("")
        res.stables = self.stables[:]
        res.outputs = self.outputs[:]
        res.offsets = self.offsets[:]
        res.stables[0] += chars
        res.outputs[0] += chars
        res.offsets[0] += len(chars)
        return res

    def __str__(self):
#        if self.error:
#            return "*"+self.outputs[-1]+"*"         # how we show an error state
        return self.outputs[-1]

    def index(self, name='base'):
        return self.slotnames[name]

    def len(self, name='simple'):
        '''Return length of input to this layer'''
        return len(self.outputs[self.slotnames[name]-1])

    def input(self, name='simple'):
        '''Return full input string to this layer'''
        return self.outputs[self.slotnames[name]-1]

    def offset(self, name='simple'):
        '''Returns the offset into input string that isn't in stables'''
        return self.offsets[self.slotnames[name]]

    def reset_output(self, name):
        '''Prepare output based on stables ready for more output to be added'''
        ind = self.index(name)
        self.outputs[ind] = self.stables[ind]

    def results(self, name, length, res):
        '''Remove from input, in effect, and put result into stables'''
        ind = self.index(name)
        leftlen = len(self.outputs[ind-1]) - self.offsets[ind] - length
        prevleft = len(self.outputs[ind-2]) - self.offsets[ind-1] if ind > 1 else 0
        self.outputs[ind] += res
        # only add to stables if everything to be consumed is already in the stables
        # of the previous layer. Otherwise, our results can only be temporary.
        if leftlen > prevleft:
            self.stables[ind] += res
            self.offsets[ind] += length

    def replace_end(self, name, length, res):
        ind = self.index(name)
        newlen = len(res)
        newstart = len(self.outputs[ind]) - newlen
        self.outputs[ind] = self.outputs[ind][:-length] + res
        diff = self.offsets[ind] - newstart
        if diff > 0:
            self.stables[ind] = self.stables[ind][:-diff]
            self.offsets[ind] = newstart

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
