subdirs = easy-chakma \
        gff-tigrinya-eritrea \
        kh-sil-angkhor \
        my-shan \
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
	@for d in $(subdirs); do echo $$d; $(MAKE) -s -C $$d; done
