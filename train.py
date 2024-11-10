
# Import các thư viện cần thiết
import os
from argparse import ArgumentParser
import tensorflow as tf
import logging
from data import NMTDataset
from transformer.model import Transformer
from transformer.optimizer import CustomLearningRate
from transformer.loss import loss_function
from trainer import Trainer

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Khởi tạo ArgumentParser để xử lý các tham số dòng lệnh
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    
    # Thêm các tham số cần thiết cho việc huấn luyện mô hình
    parser.add_argument("--input-lang", default='en', type=str, required=True)
    parser.add_argument("--target-lang", default='vi', type=str, required=True)
    parser.add_argument("--input-path", default='{}/data/train/train.en'.format(home_dir), type=str, required=True)
    parser.add_argument("--target-path", default='{}/data/train/train.vi'.format(home_dir), type=str, required=True)
    parser.add_argument("--vocab-folder", default='{}/saved_vocab/transformer/'.format(home_dir), type=str)
    parser.add_argument("--checkpoint-folder", default='{}/checkpoints/'.format(home_dir), type=str)
    parser.add_argument("--buffer-size", default=64, type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--max-length", default=40, type=int)
    parser.add_argument("--num-examples", default=1000000, type=int)
    parser.add_argument("--d-model", default=512, type=int)
    parser.add_argument("--n", default=6, type=int)
    parser.add_argument("--h", default=8, type=int)
    parser.add_argument("--d-ff", default=2048, type=int)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--dropout-rate", default=0.1, type=float)
    parser.add_argument("--eps", default=0.1, type=float)

    # Phân tích các tham số dòng lệnh
    args = parser.parse_args()

    # In thông tin chào mừng và các tham số
    print('---------------------Welcome to AI4P Transfomer-------------------')
    print('Github: gstearmit')
    print('Email: gstearmit@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Transfomer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # Khởi tạo dataset
    nmtdataset = NMTDataset(args.input_lang, args.target_lang, args.vocab_folder)
    train_dataset, val_dataset = nmtdataset.build_dataset(args.input_path, args.target_path, args.buffer_size, args.batch_size, args.max_length, args.num_examples)

    inp_tokenizer, targ_tokenizer = nmtdataset.inp_tokenizer, nmtdataset.targ_tokenizer

    # Tạo optimizer tùy chỉnh
    lrate = CustomLearningRate(args.d_model)
    optimizer = tf.keras.optimizers.Adam(lrate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Xác định kích thước từ vựng cho ngôn ngữ đầu vào và đầu ra
    inp_vocab_size = len(inp_tokenizer.word_counts) + 1
    targ_vocab_size = len(targ_tokenizer.word_counts) + 1

    # Thiết lập thư mục checkpoint
    checkpoint_folder = args.checkpoint_folder

    # Khởi tạo mô hình Transformer
    transformer = Transformer(  
        args.n, 
        args.h, 
        inp_vocab_size, 
        targ_vocab_size, 
        args.d_model, 
        args.d_ff, 
        args.activation,
        args.dropout_rate,
        args.eps
    )

    # Khởi tạo trainer
    trainer = Trainer(transformer, optimizer, args.epochs, checkpoint_folder)

    # Huấn luyện mô hình
    trainer.fit(train_dataset)
    
    # Lưu mô hình (đã được comment out)
    # transformer.save_weights(args.model_folder)


    # Command run :
    # python train.py --epochs 10 --input-lang en --target-lang vi --input-path ./data/mock/train.en --target-path ./data/mock/train.vi