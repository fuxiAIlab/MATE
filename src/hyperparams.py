# -*- coding: utf-8 -*-
# /usr/bin/python2

class Hyperparams:
    # training
    inter_op_parallelism_threads = 5
    intra_op_parallelism_threads = 5
    per_process_gpu_memory_fraction = 1

    # #######################mimic mimic mimic mimic#################
    # output_unit=76
    # # training
    # batch_size = 4
    # learning_rate = 0.001
    # maxlen = 3
    # FILE_PATH = 'data/mimic/'
    # epochs = 50
    # train_size = 440
    # test_size = 135
    # display_step = 4

    # #######################mimic_fold mimic_fold mimic_fold mimic_fold#################
    # output_unit=76
    # # training
    # batch_size = 4
    # learning_rate = 0.001
    # maxlen = 3
    # FILE_PATH = 'data/mimic_fold/'
    # epochs = 30

    # ########################meme meme meme meme meme meme#################
    # output_unit=5000
    # vocab_size = 10000
    # batch_size = 64
    # learning_rate = 0.0016
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/data_meme/'
    # epochs = 30
    # train_size=225984
    # test_size=6374

    # #######################reweet reweet reweet reweet reweet#################
    # output_unit = 3
    # batch_size = 128
    # learning_rate = 0.002
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/data_reweet/'
    # epochs = 30
    # train_size = 70133
    # test_size = 30019

    # #######################so so so so#################
    # output_unit=23
    # batch_size = 32
    # learning_rate = 0.001
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/so/'
    # epochs = 10
    # # train_size = 332683
    # # 367260
    # # 300182
    # train_size = 367260
    # # test_size = 121199
    # # 93255
    # # 107269
    # test_size = 93255

    # #######################finance finance finance finance#################
    # output_unit=3
    # batch_size = 16
    # learning_rate = 0.001
    # maxlen = 3  # max length of Pad Sequence
    # FILE_PATH = 'data/finance/'
    # epochs = 20
    # # train_size = 278592
    # # test_size = 119424
    # train_size = 331600
    # test_size = 82600
    # display_step = 4


    # # #######################nsh new user map prediction process #################
    process_type = 1
    log_file = 'log/'
    model_file = 'model/psa/'  # 模型存储文件

    output_unit = 810  # 所有地图id多分类, 类别0不参与训练
    output_sub_unit = 907
    batch_size = 2048  # 4096
    learning_rate = 0.003
    avg_time_gap = 217.54
    max_time_span = 7200.0
    fix_time_span = False
    maxlen = 256  # max length of Pad Sequence
    lastlen = 10
    dense_factor = 40  # the dense interpolation factor
    # FILE_PATH = '/data/data_preprocess/2019-06-09_map_0_10_RandomTest/'
    # TEST_FILE_PATH = '/data/data_preprocess/2019-06-09_map_0_10_RandomTest/'
    FILE_PATH = '/data/nsh_mappreload/data_source/2019-08-11/0/'
    TEST_FILE_PATH = '/data/nsh_mappreload/data_source/2019-08-18/1/'
    epochs = 30
    train_size = 848373
    test_size = 411716
    suffix_train = ''
    suffix_test = ''
    time_pred_loss_factor = 0.75
    # model parameter
    vocab_size = 810
    user_vocab_size = 1261
    min_cnt = 3  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 32
    num_blocks = 1  # number of encoder/decoder blocks
    num_decoder_blocks = 2
    num_encoder_blocks = 1
    num_heads = 8
    dropout_rate = 0.2
    portrait_vec_len = 67 + 7 * 24
    embedding_attention_size = 350
    daily_period_len = 6
    weekly_period_len = 4
    shift_hour = 6

    # # # #######################Tsinghua AppUsage user app prediction #################
    # process_type = 1
    # log_file = 'log/'
    # model_file = 'model/psa/'  # 模型存储文件
    #
    # output_unit = 1605  # 所有地图id多分类, 类别0不参与训练
    # output_sub_unit = 907
    # batch_size = 2048  # 4096
    # learning_rate = 0.003
    # avg_time_gap = 217.54
    # max_time_span = 7200.0
    # fix_time_span = False
    # maxlen = 64  # max length of Pad Sequence
    # lastlen = 5
    # dense_factor = 40  # the dense interpolation factor
    # # FILE_PATH = '/data/data_preprocess/2019-06-09_map_0_10_RandomTest/'
    # # TEST_FILE_PATH = '/data/data_preprocess/2019-06-09_map_0_10_RandomTest/'
    # FILE_PATH = '/data/TsingHuaAppUsage/space-5/train/'
    # TEST_FILE_PATH = '/data/TsingHuaAppUsage/space-5/test/'
    # epochs = 30
    # train_size = 331310
    # test_size = 68107
    # suffix_train = ''
    # suffix_test = ''
    # time_pred_loss_factor = 0.75
    # # model parameter
    # vocab_size = 1605
    # user_vocab_size = 1261
    # min_cnt = 3  # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 32
    # num_blocks = 1  # number of encoder/decoder blocks
    # num_decoder_blocks = 2
    # num_encoder_blocks = 1
    # num_heads = 8
    # dropout_rate = 0.2
    # portrait_vec_len = 24
    # embedding_attention_size = 350
    # daily_period_len = 1
    # weekly_period_len = 0
    # shift_hour = 0

    # X_file_train = FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    # Time_file_train = FILE_PATH + 'all_timetrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    # Time_file_train_label = FILE_PATH + 'all_timetrain_label_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    #
    # y_file_train = FILE_PATH + 'all_labeltrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    # X_file_test = FILE_PATH + 'all_datatest_seq' + str(maxlen + 1) + suffix_test + '.pkl'
    # Time_file_test = FILE_PATH + 'all_timetest_seq' + str(maxlen + 1) + suffix_test + '.pkl'
    # Time_file_test_label = FILE_PATH + 'all_timetest_label_seq' + str(maxlen + 1) + suffix_test + '.pkl'
    # y_file_test = FILE_PATH + 'all_labeltest_seq' + str(maxlen + 1) + suffix_test + '.pkl'
    # portrait_train = FILE_PATH + 'all_portrait_train_seq' + str(maxlen + 1) + '.pkl'
    # portrait_test = FILE_PATH + 'all_portrait_test_seq' + str(maxlen + 1) + '.pkl'
    # coordinate_seq_train = FILE_PATH + 'all_coordinatetrain_seq' + str(maxlen + 1) + '.pkl'
    # coordinate_label_train = FILE_PATH + 'all_coordinatetrain_label_seq' + str(maxlen + 1) + '.pkl'
    # coordinate_seq_test = FILE_PATH + 'all_coordinatetest_seq' + str(maxlen + 1) + '.pkl'
    # coordinate_label_test = FILE_PATH + 'all_coordinatetest_label_seq' + str(maxlen + 1) + '.pkl'
    # role_id_train = FILE_PATH + 'all_role_id_train_seq' + str(maxlen + 1) + '.pkl'
    # role_id_test = FILE_PATH + 'all_role_id_test_seq' + str(maxlen + 1) + '.pkl'
    periodic_dict = FILE_PATH + 'all_role_id_period_dict.pkl'

    X_file_train = FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    Time_file_train = FILE_PATH + 'all_timetrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    Time_gap_file_train = Time_file_train  # FILE_PATH + 'all_timetrain_gap_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    Time_file_train_label = FILE_PATH + 'all_timetrain_label_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    # Time_file_train_raw = FILE_PATH + 'all_timetrain_raw_seq' + str(maxlen + 1) + '.pkl'
    y_file_train = FILE_PATH + 'all_labeltrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'

    X_file_test = TEST_FILE_PATH + 'all_datatrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    Time_file_test = TEST_FILE_PATH + 'all_timetrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    Time_gap_file_test = Time_file_test  # FILE_PATH + 'all_timetest_gap_seq' + str(maxlen + 1) + suffix_test + '.pkl'
    Time_file_test_label = TEST_FILE_PATH + 'all_timetrain_label_seq' + str(maxlen + 1) + suffix_train + '.pkl'
    # Time_file_test_raw = FILE_PATH + 'all_timetest_raw_seq' + str(maxlen + 1) + '.pkl'
    y_file_test = TEST_FILE_PATH + 'all_labeltrain_seq' + str(maxlen + 1) + suffix_train + '.pkl'

    portrait_train = FILE_PATH + 'all_portrait_train_seq' + str(maxlen + 1) + '.pkl'
    portrait_test = TEST_FILE_PATH + 'all_portrait_train_seq' + str(maxlen + 1) + '.pkl'

    coordinate_seq_train = FILE_PATH + 'all_coordinatetrain_seq' + str(maxlen + 1) + '.pkl'
    coordinate_label_train = FILE_PATH + 'all_coordinatetrain_label_seq' + str(maxlen + 1) + '.pkl'

    coordinate_seq_test = TEST_FILE_PATH + 'all_coordinatetrain_seq' + str(maxlen + 1) + '.pkl'
    coordinate_label_test = TEST_FILE_PATH + 'all_coordinatetrain_label_seq' + str(maxlen + 1) + '.pkl'

    role_id_train = FILE_PATH + 'all_role_id_train_seq' + str(maxlen + 1) + '.pkl'
    role_id_test = TEST_FILE_PATH + 'all_role_id_train_seq' + str(maxlen + 1) + '.pkl'





