KEYMAN=sil_bengali_phonetic.kmn
LDML=sil_bengali_phonetic.ldml
REORDER=
BASE=shared/en-us-win.xml
KBDTEST=../../../../python/scripts/kbdtest
TESTS=test1.xml
LOCALE=bn-Beng
NAME="SIL Bengali Phonetic"

all: tests

$(LDML): $(KEYMAN) Makefile
	- keyman2ldml -k $< -b $(BASE) -L $(LOCALE) -N "$(NAME)" $@

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(KBDTEST) -F -t $(LDML) $t) 
