# example for metonymy resolution
python run_metonymy_resolution.py \
--data_dir ../data \
--train_file companies_train.json \
--predict_file companies_test.json \
--output_dir ../output \
--do_train  \
--do_eval \
--do_mask


# example for bert tagger (geoparsing)
python run_bert_tagger.py \
--data_dir ../data/geoparsing/gold \
--model_size large \
--output_dir ../output/geoparsing/ \
--labels ../data/geoparsing/labels_bme.txt \
--train_file train0_bme.txt \
--test_file test0_bme.txt \
--do_train \
--do_eval \
--do_predict
