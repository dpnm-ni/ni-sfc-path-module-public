import connexion
import six

from server import util
import sfc_path_selection as sfc
import sfc_using_dqn as sfc_dqn
import threading
from server.models.sfc_info import SFCInfo, SFCInfo_Custom, SFCInfo_DQN

# Determine an SFC path by Q-learning
def q_learning_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc.q_based_sfc(body)

    return response

# Determine an SFC path randomly
def random_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc.random_sfc(body)

    return response

# Determine an SFC path in which each VNF type has multiple instances
def custom_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo_Custom.from_dict(connexion.request.get_json())
        response = sfc.custom_sfc(body)

    return response

# Determine an SFC path by Deep Q-network (DQN)
def dqn_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc_dqn.dqn_based_sfc(body)

    return response

# List DQN training threads
def get_training_process():
    response = sfc_dqn.training_list

    return response

# Remove one of the running DQN training processes
def del_dqn_training(id):
    if id in sfc_dqn.training_list:
        sfc_dqn.training_list.remove(id)
        response = "Delete " + id
    else:
        response = "Fail to match a training id: " + id

    return response

# Create a thread for training a DQN model
def dqn_training(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())

        threading.Thread(target=sfc_dqn.dqn_training, args=(body,)).start()

    return "Training start!"
