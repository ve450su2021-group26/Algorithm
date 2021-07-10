# coding=utf-8
from main import main
from BertBasic import args


if __name__ == "__main__":


    label_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    data_dir = "datasets/yahoo_answers_csv"
    output_dir = ".yahoo_answers_output/"
    cache_dir = ".yahoo_answers_cache"
    log_dir = ".yahoo_answers_log/"

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    # bert-base
    bert_vocab_file = "./parameters/bert-base-uncased-vocab.txt"
    bert_model_dir = "./parameters/bert-base-uncased"

    # # bert-large
    # bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased-vocab.txt"
    # bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased"

    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)

    main(config, config.save_name, label_list)
