KEYMAN=sil_bengali_phonetic.kmn
LDML=sil_bengali_phonetic.ldml
REORDER=
BASE=shared/en-us-win.xml
KBDTEST=../../../../python/scripts/kbdtest
TESTS=test1.xml

all: tests

$(LDML): $(KEYMAN)
	- keyman2ldml -k $< -b $(BASE) $@

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(KBDTEST) -F -t $(LDML) $t) 
