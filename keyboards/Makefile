subdirs = easy-chakma \
        gff-tigrinua-eritrea \
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
	@for d in $(subdirs); do $(MAKE) -C $$d; done
