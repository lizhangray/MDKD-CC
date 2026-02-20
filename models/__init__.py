from .p2pnet_v17_lite2_repvgg2 import build17_lite2_repvgg2
from .p2pnet_v17_KD2_repvgg2_hinton_SP import build17_KD2_repvgg2_hinton_SP
from .p2pnet_v25 import build25

# build the P2PNet model
# set training to 'True' during training
def build_model17_lite2_repvgg2(args, training=False, deploy=False):
    return build17_lite2_repvgg2(args, training, deploy)

def build_model17_KD2_repvgg2_hinton_SP(args, training=False):
    return build17_KD2_repvgg2_hinton_SP(args, training)

def build_model25(args, training=False):
    return build25(args, training)