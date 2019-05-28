QUIET=t
subdirs = balochi-phonetic \
        easy-chakma \
        gff-tigrinya-eritrea \
        kh-sil-angkhor \
        my-shan \
        my-sil-visualorder1 \
        panafrica_us \
        sil-bengali \
        sil-boonkit \
        sil-devanagari-phonetic \
        sil-ethiopic-powerg \
        sil-lepcha \
        sil-mywinext \
        sil-vai \
        sil-yi \
        tainua

.PHONY: all

all:
	@for d in $(subdirs); do echo -n "$$d: "; $(MAKE) QUIET="$(QUIET)" -s -C $$d; echo ""; done
