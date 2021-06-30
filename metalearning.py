# %%
import os, sys, pickle, signal, traceback, copy, json, logging, argparse
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from tqdm import tqdm
from termcolor import colored
from munch import Munch

from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split

import pandas as pd

import preprocessor as twp

import torch
import numpy as np

import code.embedding.factory as ebd
import code.classifier.factory as clf
import code.dataset.loader as loader
import code.train.factory as train_utils



## Global Variables
CODE_ARGS = Munch({
	
	'runname': 'noname',
	
	# task configuration
	'way': 6, 
	'shot': 5,
	'query': 30, # (5 * way) 
	'dataset': "goemotions_meta",
	'n_train_class': 12, 
	'n_val_class': 11, 
	'n_test_class': 5,
	'data_path': 'data/goemotions_TweetTokenizer_lower.json',
	'wv_path':"data/", 
	'word_vector':"wiki-news-300d-1M.vec", 
	'finetune_ebd': False,
	'mode':'train', # train test finetune finetune2 supervised
	
	# load bert embeddings for sent-level datasets (optional)
	'bert':False,
	'n_workers': 20,
	'bert_cache_dir': '~/.cache/torch/transformers/', # adjust to your config (this is the default dir on linux) 
	'pretrained_bert': 'bert-base-uncased',
	
	# model options
	'auxiliary':[],
	'embedding':'transfo', # [avg, meta, cnn, transfo] 'ebd' and bert true to use bert embeddings directly
	'classifier': 'proto', # [proto, mlp, r2d2] 
	
	# distributional signature options (to apply Bao et al. 2020)
	'meta_iwf': False,
	'meta_w_target': False,
	'meta_idf': False,
	'meta_w_target_lam': 1,
	'meta_target_entropy': False,
	'meta_ebd': False,
	
	# cnn config
	'cnn_filter_sizes':[3,4,5],
	'cnn_num_filters': 50,
	
	# proto config
	'proto_hidden': [300,300],
	
	# training options 
	'lr': 1e-3,# from hyperparams tests : automatically swap to 1e-4 for Transformers
	'clip_grad': None,
	'save': True,
	'snapshot': '',
	'notqdm': False,
	'result_path': '',
	'seed': 330, 
	'dropout': 0.1,
	'patience': 20,
	'cuda': 0, # -1 for cpu,
	'scheduler': False, # learning rate scheduler did not help but can be activated here
	
	# train/test configuration
	'train_epochs': 1000, # Snell et al. 2017 used 10000 but earlystopping shows this is too much for our tests
	'train_episodes': 100,
	'val_episodes': 100,
	'test_episodes': 1000,
	
	# settings for finetuning baseline
	'finetune_loss_type': 'softmax',
	'finetune_maxepochs': 5000,
	'finetune_episodes': 10,
	'finetune_split': 0.8,
	# settings for finetuning goemotions meta models
	'finetune2_episodes': 10,
	'finetune2_maxepochs': 1,
	# target test in order to import diff vocab and model from finetuned and apply on target	
	'finetuned_dataset': 'goemotions_meta', 
	'finetuned_data_path':  'data/goemotions_TweetTokenizer_lower.json',
	'finetuned_n_train_class': 11,
	'finetuned_n_val_class': 10,
	'finetuned_n_test_class': 6,
	
	# transformer encoder configurations
	'transfo_emsize': 300,
	'transfo_nhid': 300,
	'transfo_nhead': 2,
	'transfo_nlayers': 1,
	'transfo_dropout': 0.2,
	'transfo_pe_dropout': 0.1,
	
})


# %%

class datasetCreator():
	'''
	Just a class to bundle dataset creation methods used once to create datasets available in the data folder
	'''
	def __init__(self) -> None:
		pass
	
	@staticmethod
	def creaGoEmotions2BaoJson(tokenizer='simple'):
		'''
		tokenizer: simple simple_lower TweetTokenizer
		'''
		if tokenizer in ['TweetTokenizer', 'TweetTokenizer_lower']: tweet_tokenizer = TweetTokenizer()
		twp.set_options(twp.OPT.URL, twp.OPT.RESERVED, twp.OPT.MENTION )
		label_dict = {'admiration': 1,'amusement': 2,'anger': 3,'annoyance': 4,'approval': 5,'caring': 6,'confusion': 7,'curiosity': 8,'desire': 9,'disappointment': 10,'disapproval': 11,'disgust': 12,'embarrassment': 13,'excitement': 14,'fear': 15,'gratitude': 16,'grief': 17,'joy': 18,'love': 19,'nervousness': 20,'optimism': 21,'pride': 22,'realization': 23,'relief': 24,'remorse': 25,'sadness': 26,'surprise': 27,'neutral': 28} # fix it with -1 (from 0 to 27)

		basedir = 'data/goemotions/data/'
		fplist = ['train.tsv', 'dev.tsv', 'test.tsv']
		dflist = [  pd.read_csv(basedir+filename, delimiter='\t', header=None, usecols=[0,1,2], names=['raw','label','id'])  for filename in fplist ]
		for df in dflist: print(len(df))
		dfall = pd.concat(dflist)
		print(len(dfall), 'rows in total')
		data = dfall.to_dict('records')
		print(len(data))

		def creaJsonLine(row):
			raw = twp.clean(row['raw'])
			if tokenizer == 'TweetTokenizer': text = tweet_tokenizer.tokenize(raw)
			elif tokenizer in ['TweetTokenizer_lower']: text = tweet_tokenizer.tokenize(raw.lower())
			elif tokenizer in ['simple_lower']: text = raw.lower().split(' ')
			else: text = raw.split(' ')
			labels = row['label'].split(',')
			return [{'text': text, 'raw': raw, 'label':int(label), 'id':row['id']} for label in labels ]

		jsonList = [ line for row in tqdm(data, total=len(data)) for line in creaJsonLine(row)] #flatten
		jsonLines = [json.dumps(line) for line in jsonList]
		with open('data/goemotions_{}.json'.format(tokenizer), 'w') as f: f.write('\n'.join(jsonLines))

	@staticmethod
	def creaMergedDailyDialog():
		splits = {'train':'data/ijcnlp_dailydialog/train/dailydialog_utterances_train.json', 'test':'data/ijcnlp_dailydialog/test/dailydialog_utterances_test.json', 'val': 'data/ijcnlp_dailydialog/validation/dailydialog_utterances_validation.json'}
		def addSplit(row, split): 
			row = json.loads(row)
			row['split'] = split
			return row
		records = [ addSplit(row, k) for k,v in tqdm(splits.items()) for row in open(v, 'r').read().split('\n') ]
		jsonLines = [json.dumps(line) for line  in tqdm(records)]
		with open('data/dailydialog_utterances_splits.json', 'w') as f: f.write('\n'.join(jsonLines))

	@staticmethod
	def creaGoEmotions_DailyDialogTestset(mode='highlevel'):
		"""
		creates a high level corpus dedicated to train supervised models on goemotions train and val sets and test it on dailydialog testset

		Args:
			mode: 'highlevel' to use goemotions labels mapping to Ekman or 'filter' to only use elements directly labeled with Ekman 
		
		Returns:
			nothing, creates corpus in dedicated filepath
		"""	
		mapping = {
			"anger": ["anger", "annoyance", "disapproval"],
			"disgust": ["disgust"],
			"fear": ["fear", "nervousness"],
			"joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
			"sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
			"surprise": ["surprise", "realization", "confusion", "curiosity"],
			"neutral": ["neutral"]
		}
		inverse_mapping = { el:k for k, v in mapping.items() for el in v}
		label_dict = {
			'admiration': 0,
			'amusement': 1,
			'anger': 2, #ekman
			'annoyance': 3,
			'approval': 4,
			'caring': 5,
			'confusion': 6,
			'curiosity': 7,
			'desire': 8,
			'disappointment': 9,
			'disapproval': 10,
			'disgust': 11, #ekman
			'embarrassment': 12,
			'excitement': 13,
			'fear': 14, #ekman
			'gratitude': 15,
			'grief': 16,
			'joy': 17, #ekman
			'love': 18,
			'nervousness': 19,
			'optimism': 20,
			'pride': 21,
			'realization': 22,
			'relief': 23,
			'remorse': 24,
			'sadness': 25, #ekman
			'surprise': 26, #ekman
			'neutral': 27
		}
		inverse_mapping_indices = { label_dict[k] : label_dict[v] for k,v in inverse_mapping.items() }
		ekman_indices = [2, 11, 14, 17, 25, 26]
		
		with open('data/goemotions_TweetTokenizer_lower.json', 'r') as f: messages = f.read().split('\n')
		messages = [ json.loads(m) for m in messages]
		def swapLabel(message):
			'''
			swap labels by mapping them to their ekman high level label
			'''
			message['label'] = inverse_mapping_indices[message['label']]
			return message
		if mode == 'highlevel': 
			messages = [ swapLabel(m) for m in tqdm(messages, total=len(messages)) ]
		elif mode == 'filter': 
			messages = [ m for m in tqdm(messages) if m['label'] in ekman_indices ]
		else: raise("mode should be one of the expected values: ['highlevel', 'filter'] ")

		# assign splits by message id
		with open('../data/goemotions/data/dev.tsv', 'r') as f: val_indices = f.read().split('\n')
		val_indices = [ line.split('\t')[2] for line in val_indices if len(line) > 0]
		with open('../data/goemotions/data/train.tsv', 'r') as f: train_indices = f.read().split('\n')
		train_indices = [ line.split('\t')[2] for line in train_indices if len(line) > 0]
		def assignSplit(message):
			if message['id'] in train_indices: message["split"] = "train"
			elif message['id'] in val_indices: message["split"] = "val"
			else: message["split"] = "test_ignore"
			return message
		messages = [assignSplit(m) for m in tqdm(messages, 'assigning labels')]
		messages = [m for m in tqdm(messages, desc='filtering') if m['split'] in ['train', 'val']]

		# use dailydialog testset as test
		with open('data/dailydialog_utterances_splits.json', 'r') as f: dailydialog_messages = f.read().split('\n')
		dailydialog_messages = [ json.loads(m) for m in dailydialog_messages]
		test_data = [ m for m in tqdm(dailydialog_messages, 'assigning test from dailydialog') if m["split"] == 'test']

		full_data = messages + test_data

		jsonLines = [json.dumps(line) for line in full_data]
		with open('data/goemotions_TweetTokenizer_lower_{}_testDailyDialog.json'.format(mode), 'w') as f: f.write('\n'.join(jsonLines))

def parse_args():
	parser = argparse.ArgumentParser(description="Meta-learning: Leveraging a Social Network Annotated Data Set for the Classification of Dialog Utterances into Previously Unseen Emotional Categories")

	parser.add_argument("--task", type=str, default="metalearning",
						help="Classification task"
							  "Options: [metalearning, supervised_dailydialog, supervised_goemotions_on_dailydialog]"
							  "[Default: metalearning]")
	parser.add_argument("--pipeline", type=str, default="train_test",
						help="pipeline of runs. train_test = train then test"
							  "Options: [train, test, finetune, train_test, train_finetune_test]"
							  "[Default: train_test]")
	parser.add_argument("--encoder", type=str, default="transfo",
						help="Encoder when applied. 'avg' is the faster, 'transfo' yields best results "
							  "Options: [transfo, cnn, avg]"
							  "[Default: transfo]")
	parser.add_argument("--nosave", action="store_true", default=False, help="do not save the model")
	parser.add_argument("--cuda", type=int, default=-1, help="cuda device, -1 for cpu")

	return parser.parse_args()

def print_args(args):
	"""
		Print arguments (only show the relevant arguments)
	"""
	print(colored("\nParameters:", "yellow") )
	for attr, value in sorted(args.__dict__.items()):
		if args.classifier != "proto" and attr[:6] == "proto_":
			continue
		if args.embedding != "meta" and attr[:5] == "meta_":
			continue
		if args.embedding != "cnn" and attr[:4] == "cnn_":
			continue
		if args.classifier != "mlp" and attr[:4] == "mlp_":
			continue
		if args.classifier != "transfo" and attr[:4] == "transfo_":
			continue
		print( colored( "\t{}={}".format(attr.upper(), value), "yellow" ) )

def set_seed(seed):
	"""
		Setting random seeds
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

def main(args):
	print(colored(args, 'yellow'))
	print_args(args)

	set_seed(args.seed)

	# load data
	train_data, val_data, test_data, vocab = loader.load_dataset(args)
	if args.mode == "test_from_other": 
		args_temp = copy.deepcopy(args)
		args_temp['dataset'] = args_temp['finetuned_dataset']
		args_temp['data_path'] = args_temp['finetuned_data_path']
		args_temp['n_train_class'] = args_temp['finetuned_n_train_class']
		args_temp['n_val_class'] = args_temp['finetuned_n_val_class']
		args_temp['n_test_class'] = args_temp['finetuned_n_test_class']
		train_data, val_data, test_data, vocab = loader.load_dataset(args_temp)


	# initialize model
	model = {}
	model["ebd"] = ebd.get_embedding(vocab, args)
	print('embedding built, now starting clf.get_classifier')
	model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)

	if args.mode == "train":
		# train model on train_data, early stopping based on val_data
		train_utils.train(train_data, val_data, model, args)

	elif args.mode in ["finetune"]:
		# sample an example from each class during training
		way = args.way
		query = args.query
		shot = args.shot
		args.query = 1
		args.shot= 1
		args.way = args.n_train_class
		train_utils.train(train_data, val_data, model, args)
		# restore the original N-way K-shot setting
		args.shot = shot
		args.query = query
		args.way = way

	elif args.mode in ["finetune2"]:
		# apply training continuation only if said so (> 0)
		if args.finetune2_maxepochs > 0:
			args.train_epochs = args.finetune2_maxepochs
			args.train_episodes = args.finetune2_episodes
			train_utils.train(train_data, val_data, model, args)

	elif args.mode in ['supervised']:
		print( colored('supervised', 'yellow'), len(train_data['text']) )
		args.query = 1
		args.shot= 1
		args.way = args.n_train_class
		args.train_episodes = len(train_data['text'])
		train_utils.train(train_data, val_data, model, args)


	# # testing on validation data: only for not finetune
	# # In finetune, we combine all train and val classes and split it into train
	# # and validation examples.
	if args.mode not in ["finetune"]:
		val_acc, val_std, _, _, _, _ = train_utils.test(val_data, model, args, args.val_episodes)
	else:
		val_acc, val_std = 0, 0

	if args.mode in ['test_from_other']:
		print(colored('test_from_other', 'green'), colored(args['dataset'], 'red'), colored(args['data_path'], 'red') )
		train_data, val_data, test_data, vocab = loader.load_dataset(args)
		print( colored(args['dataset'], 'green') )

	print( colored('test_data', 'green') )
	print(test_data['label'], set(test_data['label']))

	test_acc, test_std, _, _, _, _ = train_utils.test(test_data, model, args, args.test_episodes)

	if args.result_path:
		directory = args.result_path[:args.result_path.rfind("/")]
		if not os.path.exists(directory):
			os.makedirs(directory)

		result = {
			"test_acc": test_acc,
			"test_std": test_std,
			"val_acc": val_acc,
			"val_std": val_std
		}

		for attr, value in sorted(args.__dict__.items()):
			result[attr] = value

		with open(args.result_path, "wb") as f:
			pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

def runMetaBaoSystem():
	print('='*10)
	print('GoEmotions')
	print('='*10)
	runname = 'goemotions_distributional'
	print( colored(runname,'green') )

	# bao best system
	CODE_ARGS['target_iwf'] = True
	CODE_ARGS['meta_w_target'] = True
	CODE_ARGS['classifier'] = 'r2d2' # with embedding 'meta'

	## STEP1 train goemotions using train and val sets (meta training excluding ekmans 6)
	CODE_ARGS['dataset'] = "goemotions_meta"
	CODE_ARGS['embedding'] = 'meta'
	CODE_ARGS['way'] = 6 
	CODE_ARGS['shot'] = 5
	CODE_ARGS['query'] = 30
	
	CODE_ARGS['n_train_class'] = 11
	CODE_ARGS['n_val_class'] = 10 
	CODE_ARGS['n_test_class'] = 6 
	CODE_ARGS['data_path'] = 'data/goemotions_TweetTokenizer_lower.json'
	CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}_train/best_results.pkl'.format(runname, CODE_ARGS['mode'], CODE_ARGS['embedding'])
	CODE_ARGS['scheduler'] = False

	main(CODE_ARGS)

	## STEP2 evaluate on 6 dailydialog emotions
	CODE_ARGS['way'] = 6
	CODE_ARGS['query'] = 5
	CODE_ARGS['mode'] = 'test_from_other'
	CODE_ARGS['dataset'] =  'dailydialog_u_test'
	CODE_ARGS['data_path'] = 'data/dailydialog_utterances_splits.json'
	CODE_ARGS['snapshot'] = 'saved-runs/{}_train_{}_train/best'.format(runname, CODE_ARGS['embedding'])
	CODE_ARGS['n_train_class'] = 6
	CODE_ARGS['n_val_class'] = 6
	CODE_ARGS['n_test_class'] = 6
	CODE_ARGS['finetuned_dataset'] = 'goemotions_meta'
	CODE_ARGS['finetuned_data_path'] =  'data/goemotions_TweetTokenizer_lower.json'
	CODE_ARGS['finetuned_n_train_class'] = 11
	CODE_ARGS['finetuned_n_val_class'] = 10
	CODE_ARGS['finetuned_n_test_class'] = 6 

	main(CODE_ARGS)

def runSupervisedGoemotionsOnDailydialog(mode='highlevel'):
	'''

	Args:
		mode: emotion labels representation either 'highlevel' to apply emotion labels mapping to dailydialog or 'filter' to only select 6 dailydialog emotions

	'''
	print('='*10)
	print('supervised train GOEMOTIONS -> test DAILY DIALOG')
	print('='*10)
	runname = 'goemotions_filter_on_dailydialog_test'

	CODE_ARGS['lr'] = 1e-4

	CODE_ARGS['embedding'] = 'transfo'
	CODE_ARGS['way'] = 6 
	CODE_ARGS['shot'] = 2
	CODE_ARGS['query'] = 2 
	CODE_ARGS['dataset'] = "dailydialog_u_test_from_goemotions" 
	CODE_ARGS['n_train_class'] = 6
	CODE_ARGS['n_val_class'] = 6 
	CODE_ARGS['n_test_class'] = 6 
	CODE_ARGS['data_path'] = 'data/goemotions_TweetTokenizer_lower_{}_testDailyDialog.json'.format(mode)
	
	CODE_ARGS['scheduler'] = False

	CODE_ARGS['mode'] = 'supervised'
	CODE_ARGS['mlp_hidden'] = [300,CODE_ARGS['way']]
	CODE_ARGS['classifier'] = 'mlp'

	CODE_ARGS['save'] = True
	CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}/best_results.pkl'.format(runname, CODE_ARGS['mode'], CODE_ARGS['embedding']) 

	main(CODE_ARGS)

def runSupervisedDailydialog(encoder='transfo'):
	print('='*10)
	print('DAILY DIALOG supervised splits')
	print('='*10)
	runname = 'dailydialog_tmp'

	CODE_ARGS['lr'] = 1e-4
	CODE_ARGS['embedding'] = encoder
	CODE_ARGS['way'] = 6 
	CODE_ARGS['shot'] = 2
	CODE_ARGS['query'] = 2
	CODE_ARGS['dataset'] = "dailydialog_u_test" 
	CODE_ARGS['n_train_class'] = 6
	CODE_ARGS['n_val_class'] = 6 
	CODE_ARGS['n_test_class'] = 6 
	CODE_ARGS['data_path'] = 'data/dailydialog_utterances_splits.json'
	CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}_pretrain/best_results.pkl'.format(runname, CODE_ARGS['mode'], CODE_ARGS['embedding'])

	CODE_ARGS['mode'] = 'supervised'
	CODE_ARGS['mlp_hidden'] = [300,CODE_ARGS['way']]
	CODE_ARGS['classifier'] = 'mlp'

	main(CODE_ARGS)

def runMeta(encoder='transfo', train=True, finetune=False, test=True):
	'''
	Args:
		encoder: the embedding function ['cnn', 'avg', 'transfo', 'bert']
	'''
	
	if encoder == 'meta': 
		# run the distribution signature by their dedicated function
		runMetaBaoSystem()
		exit()
	
	print('='*10)
	print('GoEmotions')
	print('='*10)
	CODE_ARGS['runname'] = 'goemotions_meta_to_dailydialog'  #'goemotions_end' # 'goemotions_10w'
	print( colored(CODE_ARGS['runname'],'green') )	
	CODE_ARGS['embedding'] = encoder


	def trainEpisodes():
		## STEP1 train on goemotions using train and val sets (meta training excluding ekmans 6)
		CODE_ARGS['dataset'] = "goemotions_meta"
		CODE_ARGS['way'] = 6 
		CODE_ARGS['shot'] = 5
		CODE_ARGS['query'] = 30 

		if encoder == 'transfo': CODE_ARGS['lr'] = 1e-4
		elif encoder == 'bert':
			CODE_ARGS['bert'] = True
			CODE_ARGS['transfo_emsize'] = 768
			CODE_ARGS['transfo_nhid'] = 768
			if CODE_ARGS['bert']: CODE_ARGS['runname'] = CODE_ARGS['runname'].split('_')[0] + '_bert'
			CODE_ARGS['embedding'] = 'ebd'
			if finetune:
				CODE_ARGS['finetune_ebd'] = True
				if CODE_ARGS['finetune_ebd'] and CODE_ARGS['embedding'] == 'ebd': CODE_ARGS['runname'].replace('bert', 'bertFT')
		
		CODE_ARGS['n_train_class'] = 11
		CODE_ARGS['n_val_class'] = 10 
		CODE_ARGS['n_test_class'] = 6 
		CODE_ARGS['data_path'] = 'data/goemotions_TweetTokenizer_lower.json'
		CODE_ARGS['save'] = True
		CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}/best_results.pkl'.format(CODE_ARGS['runname'], CODE_ARGS['mode'], CODE_ARGS['embedding'])
		CODE_ARGS['scheduler'] = False

		main(CODE_ARGS)

	def finetuneEpisodes():
		## optional finetune goemotions on 6 DailyDialog testset
		CODE_ARGS['mode'] = 'finetune2'#'finetune2'
		CODE_ARGS['snapshot'] = 'saved-runs/'+CODE_ARGS['runname']+'_train_'+CODE_ARGS['embedding']+'_pretrain/best'
		CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}_finetuned/best_results.pkl'.format(CODE_ARGS['runname'], CODE_ARGS['mode'], CODE_ARGS['embedding'])

		CODE_ARGS['finetune2_maxepochs'] = 1 # 2 for fast retraining ; 0 for only finetuning
		# CODE_ARGS['finetune_maxepochs'] = 10 #5000
		# CODE_ARGS['finetune_episodes'] = 10#10

		main(CODE_ARGS)

	def testEpisodes():
		print('='*10)
		print(colored('TEST', 'yellow'))
		print('='*10)
	
		CODE_ARGS['way'] = 6
		CODE_ARGS['query'] = 5 
		CODE_ARGS['mode'] = 'test_from_other'
		CODE_ARGS['dataset'] =  'dailydialog_u_test' #'dailydialog_u_test' # 'dailydialog_u_ekman' 
		CODE_ARGS['data_path'] = 'data/dailydialog_utterances_splits.json'
		# CODE_ARGS['snapshot'] = 'saved-runs/'+runname+'_'+CODE_ARGS['embedding']+'_pretrain/best'
		if finetune:
			CODE_ARGS['snapshot'] = 'saved-runs/{}_finetune2_{}_finetuned/best'.format(CODE_ARGS['runname'], CODE_ARGS['embedding'])
		else:
			CODE_ARGS['snapshot'] = 'saved-runs/{}_train_{}/best'.format(CODE_ARGS['runname'], CODE_ARGS['embedding'])
		CODE_ARGS['n_train_class'] = 6
		CODE_ARGS['n_val_class'] = 6
		CODE_ARGS['n_test_class'] = 6

		main(CODE_ARGS)

	if train: trainEpisodes()
	if finetune: finetuneEpisodes()
	if test: testEpisodes()
	

# %%

if __name__ == "__main__":
	try:
		args = parse_args()

		CODE_ARGS['save'] = not args.nosave		
		CODE_ARGS['cuda'] = args.cuda
		
		trainModel = 'train' in args.pipeline
		finetuneModel = 'finetune' in args.pipeline
		testModel = 'test' in args.pipeline
		if args.task == 'metalearning':
			runMeta(encoder=args.encoder, train=trainModel, finetune=finetuneModel, test=testModel)
		elif args.task == 'supervised_dailydialog':
			runSupervisedDailydialog(encoder=args.encoder)
		elif args.task == 'supervised_goemotions_on_dailydialog':
			runSupervisedGoemotionsOnDailydialog(mode="filter")
		else: raise('wrong task, please use --help')
	except Exception:
		exc_info = sys.exc_info()
		traceback.print_exception(*exc_info)
		os.killpg(0, signal.SIGKILL)

	exit(0)