KEYMAN=shn_standard.kmn
LDML=sil_shan.ldml
REORDER=sil_shan_reorder.xml
BASE=shared/en-us-win.xml
KBDTEST=../../../../python/scripts/kbdtest
TESTS=test1.xml
LOCALE=shn-Mymr
NAME="Shan (SIL)"

all: tests

$(LDML): $(KEYMAN) Makefile
	- keyman2ldml -r -R $(REORDER) -k $< -b $(BASE) -L $(LOCALE) -N $(NAME) $@

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(KBDTEST) -F -t $(LDML) $t) 
