train:
	nsml run -e train.py -a "$(ARGS)" -d lpt_nfs3 --nfs-output
	 
train_adv:
	nsml run -e train_adv.py -a "$(ARGS)" -d lpt_nfs3 --nfs-output -c 10 --memory "40G" --gpu-model "P40"
		
encode_inference_data:
	python sacreBLEU/sacrebleu.py --test-set $(TEST_DATA) --language-pair ${SRC}-${TGT} --echo src | python scripts/spm_encode.py --model $(SP_MODEL) > test.${SRC}-${TGT}.${SRC}.sp

inference:										            
	nsml run -e interactive.py -a "$(DATA) --source-lang $(SRC) --target-lang $(TGT) --path $(PATH) --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" -d lpt_nfs3 
		
get_bleu:
	grep ^H logs | cut -f3 \
		| python sacreBLEU/sacrebleu.py  --test-set $(TEST_DATA) --language-pair ${SRC}-${TGT} --smooth exp --tokenize 13a --num-refs 1  -lc 


finetune:
	CUDA_VISIBLE_DEVICES=$(CUDA) python train_adv.py \
        $(DATA) \
		--max-update 12800 \
		--ddp-backend=no_c10d \
		--arch transformer \
		--optimizer adam \
		--share-decoder-input-output-embed \
		--adam-betas '(0.9, 0.98)' \
		--lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
		--warmup-updates 4000 --warmup-init-lr '1e-07' \
		--label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
		--dropout 0.3 --weight-decay 0.0001 \
		--save-dir checkpoints/$(CHECK_DIR) \
		--max-tokens 4096 \
		--update-freq 8 \
		--no-epoch-checkpoints \
		--num_cands $(NUM_CANDS) \
		--src_pert_prob $(SRC_PERT_PROB) \
		--tgt_pert_prob $(TGT_PERT_PROB) \
		--sp_model $(SP_MODEL) \
		--reset-lr-scheduler --reset-optimizer --reset-meters \
		--restore-file $(RESTORE) \
		--skip-invalid-size-inputs-valid-test

# inference_ci:
# 	    python sacreBLEU/sacrebleu.py --test-set $(TEST_DATA) --language-pair ${SRC}-${TGT} --echo src \
# 			| python scripts/spm_encode.py --model $(SP_MODEL) \
# 			> iwslt17.test.${MODE}.${SRC}-${TGT}.${SRC}.bpe \
										            
# 		cat iwslt17.test.${MODE}.${SRC}-${TGT}.${SRC}.bpe | CUDA_VISIBLE_DEVICES=$(CUDA) fairseq-interactive $(DATA) \
# 		--source-lang $(SRC) --target-lang $(TGT) --path $(CHECK_DIR) --buffer-size 2000 --batch-size 128\
# 		--beam 4  --remove-bpe sentencepiece \
# 		> iwslt17.test.${MODE}.${SRC}-${TGT}.${TGT}.sys

# 		grep ^H iwslt17.test.${MODE}.${SRC}-${TGT}.${TGT}.sys | cut -f3 > $(OUTPUT_FILE)


# spm_encode:
# 	cat ${PERT_DATA} | python scripts/spm_encode.py --model $(SP_MODEL) \
# 			    > iwslt17.test.noise.src


# get_test:
# 	python sacreBLEU/sacrebleu.py --test-set $(TEST_DATA) --language-pair ${SRC}-en --echo src \
#     | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
#     > iwslt17.test.${SRC}-en.${SRC}.bpe

# predict:
# 	cat iwslt17.test.${SRC}-en.${SRC}.bpe \
# 		| CUDA_VISIBLE_DEVICES=$(CUDA) fairseq-interactive data-bin/iwslt17.fr.en.bpe16k/ \
# 		--source-lang fr --target-lang en \
# 		--path $(CHECK_DIR) \
# 		--buffer-size 2000 --batch-size 128\
# 		--beam 4 --remove-bpe=sentencepiece  \
# 		> iwslt17.test.fr-en.en.sys

# bleu:
# 	grep ^H iwslt17.test.fr-en.en.sys | cut -f3 \
# 		| python sacreBLEU/sacrebleu.py --test-set $(TEST_DATA) --language-pair fr-en -lc --smooth exp --tokenize 13a --num-refs 1


# test_noisy_data:
# 	cat ${PERT_DATA} | python scripts/spm_encode.py --model sentencepiece/iwslt17.fr.en.unigram16k/sentencepiece.bpe.model > iwslt17.test.noise.src

# 	cat iwslt17.test.noise.src \
# 		| CUDA_VISIBLE_DEVICES=$(CUDA) fairseq-interactive data-bin/iwslt17.fr.en.unigram16k/ \
# 		--source-lang $(SRC) --target-lang $(TGT) \
# 		--path $(CHECK_DIR) \
# 		--buffer-size 2000 --batch-size 128\
# 		--beam 4 --remove-bpe=sentencepiece  \
# 		> iwslt17.noisy_test.sys	
