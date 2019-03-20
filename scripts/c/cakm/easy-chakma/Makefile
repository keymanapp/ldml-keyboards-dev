KEYMAN=easy_chakma.kmn
LDML=easy_chakma.ldml
REORDER=easy_chakma_reorder.xml
BASE=en-us-win.xml
KBDTEST=../../../../python/scripts/kbdtest
TESTS=test1.xml

all: tests

$(LDML): $(KEYMAN)
	- keyman2ldml -r -R $(REORDER) -k $< -b $(BASE) $@

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(KBDTEST) -F -t $(LDML) $t) 
