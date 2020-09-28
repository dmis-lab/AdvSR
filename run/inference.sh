

# # cs <-> en

# make inference ARGS="cs.en.sp16k  --input test.cs-en.cs.sp --source-lang cs --target-lang en --path IWSLT15_CS_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="cs.en.sp16k  --input test.cs-en.cs.sp --source-lang cs --target-lang en --path IWSLT15_CS_EN_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="cs.en.sp16k  --input test.cs-en.cs.sp --source-lang cs --target-lang en --path IWSLT15_CS_EN_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="cs.en.sp16k  --input test.cs-en.cs.sp --source-lang cs --target-lang en --path IWSLT15_CS_EN_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# make inference ARGS="en.cs.sp16k  --input test.en-cs.en.sp --source-lang en --target-lang cs --path IWSLT15_EN_CS_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.cs.sp16k  --input test.en-cs.en.sp --source-lang en --target-lang cs --path IWSLT15_EN_CS_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.cs.sp16k  --input test.en-cs.en.sp --source-lang en --target-lang cs --path IWSLT15_EN_CS_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.cs.sp16k  --input test.en-cs.en.sp --source-lang en --target-lang cs --path IWSLT15_EN_CS_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# # vi <-> en

# make inference ARGS="vi.en.sp16k  --input test.vi-en.vi.sp --source-lang vi --target-lang en --path IWSLT15_VI_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="vi.en.sp16k  --input test.vi-en.vi.sp --source-lang vi --target-lang en --path IWSLT15_VI_EN_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="vi.en.sp16k  --input test.vi-en.vi.sp --source-lang vi --target-lang en --path IWSLT15_VI_EN_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="vi.en.sp16k  --input test.vi-en.vi.sp --source-lang vi --target-lang en --path IWSLT15_VI_EN_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# make inference ARGS="en.vi.sp16k  --input test.en-vi.en.sp --source-lang en --target-lang vi --path IWSLT15_EN_VI_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.vi.sp16k  --input test.en-vi.en.sp --source-lang en --target-lang vi --path IWSLT15_EN_VI_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.vi.sp16k  --input test.en-vi.en.sp --source-lang en --target-lang vi --path IWSLT15_EN_VI_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.vi.sp16k  --input test.en-vi.en.sp --source-lang en --target-lang vi --path IWSLT15_EN_VI_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 


# # tr <-> en

# make inference ARGS="tr.en.sp16k  --input test.tr-en.tr.sp --source-lang tr --target-lang en --path IWSLT13_TR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="tr.en.sp16k  --input test.tr-en.tr.sp --source-lang tr --target-lang en --path IWSLT13_TR_EN_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="tr.en.sp16k  --input test.tr-en.tr.sp --source-lang tr --target-lang en --path IWSLT13_TR_EN_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="tr.en.sp16k  --input test.tr-en.tr.sp --source-lang tr --target-lang en --path IWSLT13_TR_EN_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# make inference ARGS="en.tr.sp16k  --input test.en-tr.en.sp --source-lang en --target-lang tr --path IWSLT13_EN_TR_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.tr.sp16k  --input test.en-tr.en.sp --source-lang en --target-lang tr --path IWSLT13_EN_TR_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.tr.sp16k  --input test.en-tr.en.sp --source-lang en --target-lang tr --path IWSLT13_EN_TR_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.tr.sp16k  --input test.en-tr.en.sp --source-lang en --target-lang tr --path IWSLT13_EN_TR_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 


# # pl <-> en

# make inference ARGS="pl.en.sp16k  --input test.pl-en.pl.sp --source-lang pl --target-lang en --path IWSLT13_PL_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="pl.en.sp16k  --input test.pl-en.pl.sp --source-lang pl --target-lang en --path IWSLT13_PL_EN_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="pl.en.sp16k  --input test.pl-en.pl.sp --source-lang pl --target-lang en --path IWSLT13_PL_EN_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="pl.en.sp16k  --input test.pl-en.pl.sp --source-lang pl --target-lang en --path IWSLT13_PL_EN_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# make inference ARGS="en.pl.sp16k  --input test.en-pl.en.sp --source-lang en --target-lang pl --path IWSLT13_EN_PL_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.pl.sp16k  --input test.en-pl.en.sp --source-lang en --target-lang pl --path IWSLT13_EN_PL_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.pl.sp16k  --input test.en-pl.en.sp --source-lang en --target-lang pl --path IWSLT13_EN_PL_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.pl.sp16k  --input test.en-pl.en.sp --source-lang en --target-lang pl --path IWSLT13_EN_PL_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 


# # fr <-> en

# make inference ARGS="fr.en.sp16k  --input test.fr-en.fr.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="fr.en.sp16k  --input test.fr-en.fr.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="fr.en.sp16k  --input test.fr-en.fr.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="fr.en.sp16k  --input test.fr-en.fr.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# make inference ARGS="en.fr.sp16k  --input test.en-fr.en.sp --source-lang en --target-lang fr --path IWSLT17_EN_FR_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.fr.sp16k  --input test.en-fr.en.sp --source-lang en --target-lang fr --path IWSLT17_EN_FR_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.fr.sp16k  --input test.en-fr.en.sp --source-lang en --target-lang fr --path IWSLT17_EN_FR_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.fr.sp16k  --input test.en-fr.en.sp --source-lang en --target-lang fr --path IWSLT17_EN_FR_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 


# # ar <-> en

# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 

# make inference ARGS="en.ar.sp16k  --input test.en-ar.en.sp --source-lang en --target-lang ar --path IWSLT17_EN_AR_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.ar.sp16k  --input test.en-ar.en.sp --source-lang en --target-lang ar --path IWSLT17_EN_AR_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.ar.sp16k  --input test.en-ar.en.sp --source-lang en --target-lang ar --path IWSLT17_EN_AR_GRAD_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="en.ar.sp16k  --input test.en-ar.en.sp --source-lang en --target-lang ar --path IWSLT17_EN_AR_GRAD_ADV_0.33/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 



# re test

make inference ARGS="fr.en.sp16k  --input iwslt17.test_perturbed_0.1.fr-en.fr.bpe.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
make inference ARGS="fr.en.sp16k  --input iwslt17.test_perturbed_0.2.fr-en.fr.bpe.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
make inference ARGS="fr.en.sp16k  --input iwslt17.test_perturbed_0.3.fr-en.fr.bpe.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
make inference ARGS="fr.en.sp16k  --input iwslt17.test_perturbed_0.4.fr-en.fr.bpe.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
make inference ARGS="fr.en.sp16k  --input iwslt17.test_perturbed_0.5.fr-en.fr.bpe.sp --source-lang fr --target-lang en --path IWSLT17_FR_EN_ADV_0.25/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 



# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_ADV_0.25_10/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_ADV_0.33_10/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_ADV_0.2_10/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 
# make inference ARGS="ar.en.sp16k  --input test.ar-en.ar.sp --source-lang ar --target-lang en --path IWSLT17_AR_EN_ADV_0.2/checkpoint_best.pt --buffer-size 2000 --batch-size 128 --beam 4 --remove-bpe sentencepiece" 




