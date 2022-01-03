#!/bin/bash

source .venv/bin/activate

# FCN
# Train
python training/embeddings.py  --dataset mtat --model_type fcn --batch_size 16 --model_load_path models/mtat/fcn/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type fcn --batch_size 16 --model_load_path models/mtat/fcn/best_model.pth --data_path MTAT --partition valid
# Test
# python training/embeddings.py  --dataset mtat --model_type fcn --batch_size 16 --model_load_path models/mtat/fcn/best_model.pth --data_path MTAT --partition test 


# MusiCNN
# Train
python training/embeddings.py  --dataset mtat --model_type musicnn --batch_size 16 --model_load_path models/mtat/musicnn/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type musicnn --batch_size 16 --model_load_path models/mtat/musicnn/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type musicnn --batch_size 16 --model_load_path models/mtat/musicnn/best_model.pth --data_path MTAT --partition test 


# CRNN
# Train
python training/embeddings.py  --dataset mtat --model_type crnn --batch_size 16 --model_load_path models/mtat/crnn/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type crnn --batch_size 16 --model_load_path models/mtat/crnn/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type crnn --batch_size 16 --model_load_path models/mtat/crnn/best_model.pth --data_path MTAT --partition test 


# Sample
# Train
python training/embeddings.py  --dataset mtat --model_type sample --batch_size 16 --model_load_path models/mtat/sample/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type sample --batch_size 16 --model_load_path models/mtat/sample/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type sample --batch_size 16 --model_load_path models/mtat/sample/best_model.pth --data_path MTAT --partition test 


# SE
# Train
python training/embeddings.py  --dataset mtat --model_type se --batch_size 16 --model_load_path models/mtat/se/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type se --batch_size 16 --model_load_path models/mtat/se/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type se --batch_size 16 --model_load_path models/mtat/se/best_model.pth --data_path MTAT --partition test 



# Short
# Train
python training/embeddings.py  --dataset mtat --model_type short --batch_size 16 --model_load_path models/mtat/short/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type short --batch_size 16 --model_load_path models/mtat/short/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type short --batch_size 16 --model_load_path models/mtat/short/best_model.pth --data_path MTAT --partition test 



# Short-Residual
# Train
python training/embeddings.py  --dataset mtat --model_type short_res --batch_size 16 --model_load_path models/mtat/short_res/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type short_res --batch_size 16 --model_load_path models/mtat/short_res/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type short_res --batch_size 16 --model_load_path models/mtat/short_res/best_model.pth --data_path MTAT --partition test 


# Attention
# Train
python training/embeddings.py  --dataset mtat --model_type attention --batch_size 16 --model_load_path models/mtat/attention/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type attention --batch_size 16 --model_load_path models/mtat/attention/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type attention --batch_size 16 --model_load_path models/mtat/attention/best_model.pth --data_path MTAT --partition test 


# HCNN
# Train
python training/embeddings.py  --dataset mtat --model_type hcnn --batch_size 16 --model_load_path models/mtat/hcnn/best_model.pth --data_path MTAT --partition train 
# Valid
python training/embeddings.py  --dataset mtat --model_type hcnn --batch_size 16 --model_load_path models/mtat/hcnn/best_model.pth --data_path MTAT --partition valid
# Test
python training/embeddings.py  --dataset mtat --model_type hcnn --batch_size 16 --model_load_path models/mtat/hcnn/best_model.pth --data_path MTAT --partition test 