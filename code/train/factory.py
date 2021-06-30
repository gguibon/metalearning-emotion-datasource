import code.train.regular as regular
import code.train.finetune as finetune


def train(train_data, val_data, model, args):
     return regular.train(train_data, val_data, model, args)


def test(test_data, model, args, verbose=True):

    
    # if args.mode in ['finetune']:
    #     return finetune.test(test_data, model, args, verbose)
    # else:
    return regular.test(test_data, model, args, verbose, target='test')
