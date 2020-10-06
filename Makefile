preprocess:
	python ../preprocess.py \
		--source-lang cs --target-lang en \
		--trainpref ../${TEXT}/train.sp.cs-en --validpref ../${TEXT}/valid.sp.cs-en \
		--destdir ../data-bin/iwslt15.cs.en --joined-dictionary \
		--workers 4

train_adv:
	CUDA_VISIBLE_DEVICES=${CUDA} python train.py \
		${DATA} \
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
	    --save-dir checkpoints/${CHECK_DIR} \
	    --max-tokens 4096 \
	    --no-epoch-checkpoints \
		--update-freq 8 \
		--num_cands ${NUM_CANDS} \
		--src_pert_prob ${SRC_PERT_PROB} \
		--tgt_pert_prob ${TGT_PERT_PROB} \
		--sp_model ${SPM_DIR}
		--adv_sr 

inference:
	python sacreBLEU/sacrebleu.py --test-set ${TEST_DATA} --language-pair ${SRC}-${TGT} --echo src \
		| python scripts/spm_encode.py --model ${SPM_DIR} \
		> test.${SRC}-${TGT}.${SRC}.sp \
												
	cat iwslt17.test.${SRC}-${TGT}.${SRC}.sp | CUDA_VISIBLE_DEVICES=${CUDA} fairseq-interactive ${DATA} \
	--source-lang ${SRC} --target-lang ${TGT} --path ${CHECK_DIR} --buffer-size 2000 --batch-size 128\
	--beam 4  --remove-bpe sentencepiece \
	> test.${SRC}-${TGT}.${TGT}.sys

	grep ^H test.${SRC}-${TGT}.${TGT}.sys | cut -f3 \
		| python sacreBLEU/sacrebleu.py  --test-set ${TEST_DATA} --language-pair ${SRC}-${TGT} --smooth exp --tokenize 13a --num-refs 1  -lc 




