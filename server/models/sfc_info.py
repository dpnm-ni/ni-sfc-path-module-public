# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from server.models.base_model_ import Model
from server import util


class SFCInfo(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, sfcr_name: str=None, sfc_prefix: str=None, sfc_vnfs: List[str]=None, sfc_name: str=None):  # noqa: E501
        """SFCInfo - a model defined in Swagger

        :param sfcr_name: The sfcr_name of this SFCInfo.  # noqa: E501
        :type sfcr_name: str
        :param sfc_prefix: The sfc_prefix of this SFCInfo.  # noqa: E501
        :type sfcr_src_ip_prefix_name: str
        :param sfc_vnfs: The sfc_vnfs of this SFCInfo.  # noqa: E501
        :type sfc_vnfs: List[str]
        :param sfc_name: The sfc_name of this SFCInfo.  # noqa: E501
        :type sfc_name: str
        """
        self.swagger_types = {
            'sfcr_name': str,
            'sfc_prefix': str,
            'sfc_vnfs': List[str],
            'sfc_name': str
        }

        self.attribute_map = {
            'sfcr_name': 'sfcr_name',
            'sfc_prefix': 'sfc_prefix',
            'sfc_vnfs': 'sfc_vnfs',
            'sfc_name': 'sfc_name'
        }

        self._sfcr_name = sfcr_name
        self._sfc_prefix = sfc_prefix
        self._sfc_vnfs = sfc_vnfs
        self._sfc_name = sfc_name

    @classmethod
    def from_dict(cls, dikt) -> 'SFCInfo':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The SFCInfo of this SFCInfo.  # noqa: E501
        :rtype: SFCInfo
        """
        return util.deserialize_model(dikt, cls)

    @property
    def sfcr_name(self) -> str:
        """Gets the sfcr_name of this SFCInfo.


        :return: The sfcr_name of this SFCInfo.
        :rtype: str
        """
        return self._sfcr_name

    @sfcr_name.setter
    def sfcr_name(self, sfcr_name: str):
        """Sets the sfcr_name of this SFCInfo.


        :param sfcr_name: The sfcr_name of this SFCInfo.
        :type sfcr_name: str
        """

        self._sfcr_name = sfcr_name

    @property
    def sfc_vnfs(self) -> List[str]:
        """Gets the sfc_vnfs of this SFCInfo.


        :return: The sfc_vnfs of this SFCInfo.
        :rtype: List[str]
        """
        return self._sfc_vnfs

    @sfc_vnfs.setter
    def sfc_vnfs(self, sfc_vnfs: List[str]):
        """Sets the sfc_vnfs of this SFCInfo.


        :param sfc_vnfs: The sfc_vnfs of this SFCInfo.
        :type sfc_vnfs: List[str]
        """

        self._sfc_vnfs = sfc_vnfs

    @property
    def sfc_name(self) -> str:
        """Gets the sfc_name of this SFCInfo.


        :return: The sfc_name of this SFCInfo.
        :rtype: str
        """
        return self._sfc_name

    @sfc_name.setter
    def sfc_name(self, sfc_name: str):
        """Sets the sfc_name of this SFCInfo.


        :param sfc_name: The sfc_name of this SFCInfo.
        :type sfc_name: str
        """

        self._sfc_name = sfc_name

    @property
    def sfc_prefix(self) -> str:
        """Gets the sfc_prefix of this SFCInfo.


        :return: The sfc_prefix of this SFCInfo.
        :rtype: str
        """
        return self._sfc_prefix

    @sfc_prefix.setter
    def sfc_prefix(self, sfc_prefix: str):
        """Sets the sfc_prefix of this SFCInfo.


        :param sfc_prefix: The sfc_prefix of this SFCInfo.
        :type sfc_prefix: str
        """

        self._sfc_prefix = sfc_prefix
