subdirs = easy-chakma \
        kh-sil-angkhor \
        my-shan \
        sil-bengali \
        sil-boonkit \
        sil-devanagari-phonetic \
        sil-lepcha \
        sil-vai \
        sil-yi 

.PHONY: all

all:
	@for d in $(subdirs); do $(MAKE) -C $$d; done
