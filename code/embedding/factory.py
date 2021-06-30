import torch
import datetime

from code.embedding.wordebd import WORDEBD
from code.embedding.cxtebd import CXTEBD

from code.embedding.avg import AVG
from code.embedding.cnn import CNN
from code.embedding.meta import META
from code.embedding.transfo import TransformerModel as TRANSFO


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    # check if loading pre-trained embeddings
    if args.bert:
        ebd = CXTEBD(args.pretrained_bert,
                     cache_dir=args.bert_cache_dir,
                     finetune_ebd=args.finetune_ebd,
                     return_seq=(args.embedding!='ebd'))
    else:
        ebd = WORDEBD(vocab, args.finetune_ebd)

    if args.embedding == 'avg':
        model = AVG(ebd, args)
    elif args.embedding == 'cnn':
        model = CNN(ebd, args)
    elif args.embedding == "transfo":
        model = TRANSFO(ebd, args)
    elif args.embedding in ['meta', 'meta_mlp']:
        model = META(ebd, args)
    elif args.embedding == 'ebd' and args.bert:
        model = ebd  # using bert representation directly

    print("{}, Building embedding".format(datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    if args.snapshot != '':
        # load pretrained models
        print("{}, Loading pretrained embedding from {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            '{}.ebd'.format(args.snapshot)
            ))
        model.load_state_dict(  torch.load( '{}.ebd'.format(args.snapshot) ), strict=False  )

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
