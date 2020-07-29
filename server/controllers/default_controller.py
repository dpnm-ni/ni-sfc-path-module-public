import connexion
import six

from server import util
import sfc_path_selection as sfc
from server.models.sfc_info import SFCInfo  # noqa: E501

def q_learning_sfc(body):  # noqa: E501
    """sfc path selection using q-learning

     # noqa: E501


    :rtype: str
    """

    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc.q_based_sfc(body)

    return response


def random_sfc(body):  # noqa: E501
    """sfc path selection randomly

     # noqa: E501

    :rtype: str
    """
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc.random_sfc(body)

    return response
