# # cs en

for LOG_NUM in 543 544
do
    nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
    make get_bleu LOGS=logs TEST_DATA=iwslt15/tst2013 SRC=cs TGT=en
done

# # # en cs 

# for LOG_NUM in 521 # 0.33 best 23.5
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt15/tst2013 SRC=en TGT=cs
# done


# # vi en

# for LOG_NUM in 116 117 # 0.33 best 29.3
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt15/tst2013 SRC=vi TGT=en
# done

# # en vi

# for LOG_NUM in 120 121 # 0.33 best 32.4
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt15/tst2013 SRC=en TGT=vi
# done

# # tr en 

# for LOG_NUM in 134 135 # 0.33 best 24.5
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt13/tst2010 SRC=tr TGT=en
# done

# # en tr

# for LOG_NUM in 138 139 # 0.25 & 0.33 best 14.6
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt13/tst2010 SRC=en TGT=tr
# done

# # pl en

# for LOG_NUM in 142 143 # 0.25 & 0.33 best 20.6
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt13/tst2010 SRC=pl TGT=en
# done

# # en pl

# for LOG_NUM in 146 147 # 0.25 best 15.1
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt13/tst2010 SRC=en TGT=pl
# done


# for LOG_NUM in 150 151
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt17/tst2015 SRC=fr TGT=en
# done

# for LOG_NUM in 154 155
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt17/tst2015 SRC=en TGT=fr
# done

# for LOG_NUM in 532
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt17/tst2015 SRC=ar TGT=en
# done

# for LOG_NUM in 528 529
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt17/tst2015 SRC=en TGT=ar
# done



# for LOG_NUM in 522 523
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt17/tst2015 SRC=fr TGT=en
# done


# for LOG_NUM in 524 525
# do
#     nsml logs KR62623/lpt_nfs3/$LOG_NUM > logs
#     make get_bleu LOGS=logs TEST_DATA=iwslt17/tst2015 SRC=en TGT=fr
# done