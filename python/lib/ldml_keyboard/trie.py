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

from collections import namedtuple
from UnicodeSets import UnicodeSetSequence, UnicodeSet, parse

Match = namedtuple('Match', ['offset', 'length', 'rule'])

class Node(dict):
    def __init__(self, *a, **kw):
        parent = None
        if len(a) > 0:
            parent = a[0]
            a = a[1:]
        super(Node, self).__init__(*a, **kw)
        self.match = None
        self.parent = parent
        self.negmatch = None
        self.negset = set()

def filterset(seq, filterlist):
    return filter(len, (UnicodeSet([c for c in y if c in filterlist]) for y in seq))

class Trie(object):
    def __init__(self):
        self.forwards = [Node()]
        self.backwards = [Node()]
        self.singles = {}

    def append(self, fr, before, after, rule, filterlist=None, normal=None):
        frc = parse(fr, normal=normal) if fr is not None else UnicodeSetSequence()
        beforec = parse(before, normal=normal) if before is not None else UnicodeSetSequence()
        afterc = parse(after, normal=normal) if after is not None else UnicodeSetSequence()
        if filterlist is not None:
            frc = filterset(frc, filterlist)
            beforec = filterset(beforec, filterlist)
            afterc = filterset(afterc, filterlist)
        if len(frc) == 0:
            return

        if len(beforec) == 0 and len(afterc) == 0 and len(frc) == 1:
            m = Match(0, 1, rule)
            for c in frc[0]:
                if c == u'\uFDD1':
                    c = u'\u200B'
                if c in self.singles:
                    self.singles[c] = Match(m.offset, m.length, \
                                            self.singles[c].rule._newmerge(rule))
                else:
                    self.singles[c] = m
            return

        while len(self.forwards) <= len(beforec):
            self.forwards.append(Node())
        jobs = [(self.forwards[len(beforec)], None)]
        for i in range(2):
            ocount = len(beforec)
            length = len(frc)
            m = Match(ocount, length + ocount, rule)

            for cset in beforec + frc + afterc:
                newjobs = []
                newnjobs = []
                for j, n in jobs:
                    if cset.negative:
                        newnjobs.append((j, cset))
                    for c in cset:
                        if c == u'\uFDD1':
                            c = u'\u200B'
                        newjobs.append((j.setdefault(c, Node(j)), c))
                jobs = newjobs
            for j, c in jobs:
                if j.match is not None:
                    j.match = self._override(j.match, m)
                elif j.parent is not None and j.parent.negmatch is not None and c in j.parent.negset:
                    j.match = self._override(j.parent.negmatch, m)
                else:
                    j.match = m
            for j, cset in newnjobs:
                for c, sub in j.items():
                    if c in cset or sub.match is None:
                        continue
                    if sub.match.offset == ocount and sub.match.length == length + ocount:
                        sub.match = Match(ocount, length + ocount, \
                                          sub.match.rule._newmerge(rule))
                    elif sub.match.length < length:
                        sub.match = m

            temp = afterc
            while len(self.backwards) <= len(afterc):
                self.backwards.append(Node())
            afterc = list(reversed(beforec))
            beforec = list(reversed(temp))
            frc = list(reversed(frc))
            jobs = [(self.backwards[len(beforec)], None)]

    def _override(self, base, other):
        if base.offset == other.offset and base.length == other.length:
            return Match(base.offset, base.length, other.rule._newmerge(base.rule))
        elif base.length < other.length:
            return other
        else:
            return base

    def match(self, s, ind, skipbefore=False, base=None):
        if not len(s):
            return [0, 0, None, False]
        if base is None:
            base = self.forwards
        firstc = s[ind]
        last = self.singles.get(firstc, (0, 0, None))
        curlast = last
        curlen = 0
        curpfit = False
        if firstc in base[0]:
            (curlast, curlen, curpfit) = self._testmatch(s[ind:], base[0], skipbefore)
            if curlast is None:
                curlast = last
        for i, b in enumerate(base[1:]):
            if i + 1 <= ind:
                if s[ind-i-1] in b:
                    (res, reslen, respfit) = self._testmatch(s[ind-i-1:], b, skipbefore)
                    if res is not None and (res[1] - res[0] > curlast[1] - curlast[0] or \
                            (res[1]-res[0] == curlast[1]-curlast[0] and reslen >= curlen)):
                        curlast = (res.offset - i - 1, res.length - i - 1, res.rule)
                        curlen = reslen
                        curpfit = respfit
        return list(curlast) + [curpfit]

    def _testmatch(self, s, curr, skipbefore):
        last = None
        curri = 0
        for i, c in enumerate(s):
            if c not in curr:
                if curr.match is not None:
                    return (curr.match, i, len(curr) != 0)
                else:
                    return (last, curri, True)
            if curr.match is not None and (not skipbefore or curr.match.offset == 0):
                last = curr.match
                curri = i+1
            curr = curr[c]
        if curr.match is not None:
            last = curr.match
        return (last, len(s), len(curr) != 0)

    def revmatch(self, s, ind, skipafter=False):
        return self.match(list(reversed(s)), ind, base=self.backwards, skipbefore=skipafter)
