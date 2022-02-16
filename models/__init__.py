from models.bjnet_fused  import BJNet
from models.loss import model_loss

__models__ = {
    "fused": BJNet,
    "cascade": BJNet,
}
