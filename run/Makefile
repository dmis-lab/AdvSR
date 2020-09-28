train:
	nsml run -e train.py -a "$(ARGS)" -d lpt_nfs3 --nfs-output

train_adv:
	nsml run -e train_adv.py -a "$(ARGS)" -d lpt_nfs3 --nfs-output -c 5 --memory "40G" --gpu-model "P40"
		
encode_inference_data:
	python sacreBLEU/sacrebleu.py --test-set $(TEST_DATA) --language-pair ${SRC}-${TGT} --echo src | python scripts/spm_encode.py --model $(SP_MODEL) > $(TEST_DATA).sp

encode_inference_data_noisy:
	cat $(TEST_DATA) | python scripts/spm_encode.py --model $(SP_MODEL) > $(TEST_DATA).sp

inference:
	nsml run -e interactive.py -a "$(ARGS)" -d lpt_nfs3 
		
get_bleu:
	grep ^H $(LOGS) | cut -f3 \
		| python sacreBLEU/sacrebleu.py  --test-set $(TEST_DATA) --language-pair ${SRC}-${TGT} --smooth exp --tokenize 13a --num-refs 1  -lc