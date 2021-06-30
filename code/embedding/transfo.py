import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


class TransformerModel(nn.Module):
	"""Transformer Encoder Module."""

	def __init__(self, ebd, args, dropout=0.5):
		super(TransformerModel, self).__init__()
		
		self.args = args
		self.ebd = ebd
		self.ebd_dim = self.ebd.embedding_dim
		self.input_dim = self.ebd.embedding_dim # + self.aux.embedding_dim # self.aux was removed

		# ntokens = len(corpus.dictionary) ## no need for ntokens (i.e. embedding size) (we are using pretrained ones)
		self.ninp = args.transfo_emsize # == emsize # ninp number of inputs
		self.nhead = args.transfo_nhead 
		self.nhid = args.transfo_nhid
		self.nlayers = args.transfo_nlayers

		self.model_type = 'Transformer'
		self.src_mask = None
		self.pos_encoder = PositionalEncoding(self.ninp, dropout=args.transfo_pe_dropout)

		encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, args.transfo_dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
		
		self.encoder = self.ebd
		
		# self.decoder = nn.Linear(self.ninp, ntoken)
		self.decoder = nn.Linear(self.ninp, self.ebd_dim) #ebd.vocab_size)

		# self.init_weights()

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self):
		initrange = 0.1
		# nn.init.uniform_(self.encoder.weight, -initrange, initrange)
		# nn.init.zeros_(self.decoder.weight)
		# nn.init.uniform_(self.decoder.weight, -initrange, initrange)

	def forward(self, src):
		
		if self.src_mask is None or self.src_mask.size(0) != src['text'].size(0):
			device = src['text'].device
			mask = self._generate_square_subsequent_mask(src['text'].size(0)).to(device)
			self.src_mask = mask

		# # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, self.src_mask)
		output = torch.mean(output, 1)
		return output