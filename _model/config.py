from yacs.config import CfgNode as CN


_C = CN()

_C.train = CN()
# batch size
_C.train.inverse = False
_C.train.eval_only = False
_C.train.debug = True
_C.train.data_path = "/work1/maddison/haonand/train_set_mmseq10_uniref50_newtasks_testballnotremoved.csv"
_C.train.load_uniref50 = False
_C.train.batch_size = 10
_C.train.weighted_sample = False
_C.train.epochs = 1
_C.train.num_itr = 100000
_C.train.encoder_lr = 1e-4
_C.train.decoder_lr = 1e-4
_C.train.lr = 1e-4
_C.train.scheduler= False
_C.train.num_gpu = 8
_C.train.esm_dropout = 0.0
_C.train.float_precision = "bfloat16"
_C.train.freeze_encoder = False
_C.train.unfreeze_last_layer = False
_C.train.freeze_decoder = False
_C.train.num_node = 1
_C.train.encoder_name = "facebook/esm2_t30_150M_UR50D"
_C.train.decoder_name = "allenai/scibert_scivocab_uncased"
_C.train.parallel = "ddp"
_C.train.train_on_all = True
_C.train.encoder_max_len = 1024
_C.train.decoder_max_len = 128
_C.train.weight_decay = 1e-4
_C.train.hidden_size = 512
_C.train.max_grad_norm = 1
_C.train.scheduler_step_size = 5000
_C.train.scheduler_gamma = 0.9

_C.fact = CN()

_C.test = CN()
_C.test.data_path = "/work1/maddison/haonand/val_set_mmseq10_uniref50.txt"
_C.test.ec = False
_C.test.name = False
_C.test.go = False
_C.test.deeploc = False
_C.test.lm_loss = False
_C.test.num_itr = 1000
_C.test.ec_test_size = 10000
_C.test.ec_batch_size = 80
_C.test.name_test_size = 5000
_C.test.name_batch_size = 50
_C.test.go_test_size = 500
_C.test.go_batch_size =  50
_C.test.lm_batch_size = 50


_C.decode = CN()
_C.decode.ec_num_beams = 1
_C.decode.ec_temperature = 1.0
_C.decode.name_num_beams = 1
_C.decode.name_temperature = 1.0
_C.decode.go_num_beams = 1 
_C.decode.go_temperature = 1.0
_C.decode.ec_max_length = 32
_C.decode.name_max_length = 50
_C.decode.go_max_length = 256
_C.decode.sample = False
_C.decode.name_sample = False

_C.lora = CN()
_C.lora.use_lora = False
_C.lora.rank = 128
_C.lora.dropout = 0
_C.lora.alpha = 32

_C.train.save = True
_C.save_every = True
_C.save_path = "/work1/maddison/haonand/checkpoints"
_C.load_checkpoint = False
_C.load_encoder_only = False
_C.load_optimizer = False
_C.checkpoint_path = "/work1/maddison/haonand/checkpoints/protclip_23706_2024_02_19_18_15_51.pth"


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
