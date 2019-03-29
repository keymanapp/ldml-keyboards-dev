KBDTEST ?= ../../python/scripts/kbdtest

.PHONEY: all tests

all: tests

ifneq (,$(wildcard $(KEYMAN)))
$(LDML): $(KEYMAN) Makefile $(REORDER) $(IMPORT)
	- keyman2ldml $(if $(HAS_REORDER)$(REORDER),-r) $(if $(REORDER),-R $(REORDER)) -k $< $(if $(BASE),-b $(BASE)) $(if $(LOCALE),-L $(LOCALE)) $(if $(NAME),-N $(NAME)) $(if $(IMPORT),-i $(IMPORT)) $(if $(debug),-z 4) $(OPTIONS) $@
endif

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(KBDTEST) -F -t $(LDML) $t)
