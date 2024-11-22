echo "Running for synthetic dataset 1"

echo "Running for unixcoder"
python evaluation.py --model_name unixcoder --dataset_path ../data/eval_synthetic_1 --model_path models/unixcoder/checkpoint-28000 --output_filename synthetic_1/unixcoder.csv

echo "Running for unixcoder oversample"
python evaluation.py --model_name unixcoder --dataset_path ../data/eval_synthetic_1 --model_path models/unixcoder_typecheck_oversample/checkpoint-28000 --output_filename synthetic_1/unixcoder_typecheck_oversample.csv

echo "Running for codebert"
python evaluation.py --model_name codebert --dataset_path ../data/eval_synthetic_1 --model_path models/codebert/checkpoint-28000 --output_filename synthetic_1/codebert.csv

echo "Running for codebert oversample"
python evaluation.py --model_name codebert --dataset_path ../data/eval_synthetic_1 --model_path models/codebert_typecheck_oversample/checkpoint-28000 --output_filename synthetic_1/codebert_typecheck_oversample.csv

echo "Running for graphcodebert"
python evaluation.py --model_name graphcodebert --dataset_path ../data/eval_synthetic_1 --model_path models/graphcodebert/checkpoint-28000 --output_filename synthetic_1/graphcodebert.csv

echo "Running for graphcodebert oversample"
python evaluation.py --model_name graphcodebert --dataset_path ../data/eval_synthetic_1 --model_path models/graphcodebert_typecheck_oversample/checkpoint-28000 --output_filename synthetic_1/graphcodebert_typecheck_oversample.csv

echo "Running for real dataset py150"

echo "Running for unixcoder"
python evaluation.py --model_name unixcoder --dataset_path ../data/eval_real --model_path models/unixcoder/checkpoint-28000 --output_filename real/unixcoder.csv

echo "Running for unixcoder oversample"
python evaluation.py --model_name unixcoder --dataset_path ../data/eval_real --model_path models/unixcoder_typecheck_oversample/checkpoint-28000 --output_filename real/unixcoder_typecheck_oversample.csv

echo "Running for codebert"
python evaluation.py --model_name codebert --dataset_path ../data/eval_real --model_path models/codebert/checkpoint-28000 --output_filename real/codebert.csv

echo "Running for codebert oversample"
python evaluation.py --model_name codebert --dataset_path ../data/eval_real --model_path models/codebert_typecheck_oversample/checkpoint-28000 --output_filename real/codebert_typecheck_oversample.csv

echo "Running for graphcodebert"
python evaluation.py --model_name graphcodebert --dataset_path ../data/eval_real --model_path models/graphcodebert/checkpoint-28000 --output_filename real/graphcodebert.csv

echo "Running for graphcodebert oversample"
python evaluation.py --model_name graphcodebert --dataset_path ../data/eval_real --model_path models/graphcodebert_typecheck_oversample/checkpoint-28000 --output_filename real/graphcodebert_typecheck_oversample.csv


echo "Running for real dataset pypi"

echo "Running for unixcoder"
python evaluation.py --model_name unixcoder --dataset_path ../data/eval_real_2 --model_path models/unixcoder/checkpoint-28000 --output_filename real_2/unixcoder.csv

echo "Running for unixcoder oversample"
python evaluation.py --model_name unixcoder --dataset_path ../data/eval_real_2 --model_path models/unixcoder_typecheck_oversample/checkpoint-28000 --output_filename real_2/unixcoder_typecheck_oversample.csv

echo "Running for codebert"
python evaluation.py --model_name codebert --dataset_path ../data/eval_real_2 --model_path models/codebert/checkpoint-28000 --output_filename real_2/codebert.csv

echo "Running for codebert oversample"
python evaluation.py --model_name codebert --dataset_path ../data/eval_real_2 --model_path models/codebert_typecheck_oversample/checkpoint-28000 --output_filename real_2/codebert_typecheck_oversample.csv

echo "Running for graphcodebert"
python evaluation.py --model_name graphcodebert --dataset_path ../data/eval_real_2 --model_path models/graphcodebert/checkpoint-28000 --output_filename real_2/graphcodebert.csv

echo "Running for graphcodebert oversample"
python evaluation.py --model_name graphcodebert --dataset_path ../data/eval_real_2 --model_path models/graphcodebert_typecheck_oversample/checkpoint-28000 --output_filename real_2/graphcodebert_typecheck_oversample.csv

