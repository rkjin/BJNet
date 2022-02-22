from models.bjnet_fused_3rd import BJNet
from models.loss import model_loss

__models__ = {
    "fused": BJNet,
    "cascade": BJNet,
}
