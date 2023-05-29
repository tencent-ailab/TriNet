import sys
import numpy as np

###Utils for loaders###
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def printlog(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)

def splice(feats, lctx, rctx):
    length = feats.shape[0]
    dim    = feats.shape[1]

    padding = np.zeros((length + lctx + rctx, dim), dtype=np.float32)
    padding[:lctx] = feats[0]
    padding[lctx:lctx+length] = feats
    padding[lctx+length:] = feats[-1]

    spliced = np.zeros((length,  dim * (lctx + 1 + rctx)), dtype=np.float32)
    for i in range(lctx + 1 + rctx):
        spliced[:, i*dim:(i+1)*dim] = padding[i:i+length,:]
    return spliced

def putThread(queue, generator, *gen_args):
    for item in generator(*gen_args):
        queue.put(item)
        if item is None: 
            break

def getInputDim(args):
    return args.feats_dim * (args.lctx + 1 + args.rctx)
