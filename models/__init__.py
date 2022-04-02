from models.cfnet_4th import CFNet_modified
from models.cfnet_reference import CFNet 
from models.loss import model_loss

__models__ = {
    "cfnet_modified": CFNet_modified,
    "cascade": CFNet_modified,
    "cfnet":CFNet
}
