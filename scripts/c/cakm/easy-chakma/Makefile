KEYMAN=easy_chakma.kmn
LDML=easy_chakma.ldml
REORDER=easy_chakma_reorder.xml
BASE=shared/en-us-win.xml
KBDTEST=../../../../python/scripts/kbdtest
TESTS=test1.xml
LOCALE=ccp-Cakm
NAME="Easy Chakma"

all: tests

$(LDML): $(KEYMAN) Makefile
	- keyman2ldml -r -R $(REORDER) -k $< -b $(BASE) -L $(LOCALE) -N $(NAME) $@

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(KBDTEST) -F -t $(LDML) $t) 
