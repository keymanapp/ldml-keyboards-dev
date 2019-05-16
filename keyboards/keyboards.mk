KBDTEST ?= python ../../python/scripts/kbdtest

.PHONEY: all tests

define execute-test
	$(KBDTEST) -F -t $(TESTOPTIONS) $(LDML) $(1)

endef

all: tests

ifneq (,$(wildcard $(KEYMAN)))
$(LDML): $(KEYMAN) Makefile $(REORDER) $(IMPORT)
	- keyman2ldml $(if $(HAS_REORDER)$(REORDER),-r) $(if $(REORDER),-R $(REORDER)) -k $< $(if $(BASE),-b $(BASE)) $(if $(LOCALE),-L $(LOCALE)) $(if $(NAME),-N $(NAME)) $(if $(IMPORT),-i $(IMPORT)) $(if $(debug),-z 4) $(OPTIONS) $@
endif

tests: $(LDML) $(TESTS)
	- $(foreach t,$(TESTS),$(call execute-test,$t))
