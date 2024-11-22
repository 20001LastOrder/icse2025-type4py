# Run full dataset
# Codebert
python finetune.py --model_name microsoft/codebert-base --dataset_path ../data/

# GraphCodeBert
python finetune.py --model_name microsoft/graphcodebert-base --dataset_path ../data/

# UnixCoder
python finetune.py --model_name microsoft/graphcodebert-base --dataset_path ../data/

# Run dataset filtered with typecheck
# Codebert
python finetune.py --model_name microsoft/codebert-base --dataset_path ../data/ --do_typecheck

# GraphCodeBert
python finetune.py --model_name microsoft/graphcodebert-base --dataset_path ../data/ --do_typecheck

# UnixCoder
python finetune.py --model_name microsoft/graphcodebert-base --dataset_path ../data/ --do_typecheck