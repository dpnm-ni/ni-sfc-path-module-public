# ni-sfc-path-module-public

## Main Responsibilities
Random or RL-based SFC path selection module.
- Provide APIs to create random SFCs
- Provide APIs to create SFCs by using Q-learning

## Requirements
```
Python 3.5.2+
```

Please install pip3 and requirements by using the command as below.
```
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt
```

## Configuration
This module runs as web server to handle an SFC request that describes required data for an SFC.
To use a web UI of this module or send an SFC request to the module, a port number can be configured (a default port number is 8001)

```
# server/__main__.py
def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'NI SFC Sub-Module Service'})
    app.run(port=8001) ### Port number configuration
```

This module interacts with ni-mano to create SFC in OpenStack environment.
To communicate with ni-mano, this module should know URI of ni-mano.
In ni-mano, ni_mon and ni_nvfo are responsible for interacting with this module so their URI should be configured as follows.

```
# config/config.yaml
ni_mon:
  host: http://<ni_mon_ip>:<ni_mon_port>      # Configure here to interact with a monitoring module
ni_nfvo:
  host: http://<ni_nfvo_ip>:<ni_nfvo_port>    # Configure here to interact with an NFVO module
```

Before running this module, OpenStack network ID should be configured because VNF instances in OpenStack can have multiple network interfaces.
This module uses openstack_network_id value to identify a network interface used to crate an SFC.
Moreover, Q-learning hyper-parameters can be configured as follows (they have default values).

```
# sfc_path_selection.py

# Parameters
# OpenStack Parameters
openstack_network_id = ""    # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (Q-learning in this codes)
learning_rate = 0.10         # Learning rate
discount_factor = 0.60       # Discount factor
initial_epsilon = 0.90       # epsilon value of -greedy algorithm
num_episode = 3000           # Number of iteration for Q-learning
```

## Usage

After installation and configuration of this module, you can run this module by using the command as follows.

```
python3 -m server
```

This module provides web UI based on Swagger:

```
http://<host IP running this module>:<port number>/ui/
```

To create an SFC in OpenStack testbed, this module processes a HTTP POST message including SFCInfo data in its body.
You can generate an SFC request by using web UI or using other library creating HTTP messages.
If you create and send a HTTP POST message to this module, the destination URI is as follows.

```
# Chhose and create an optimal SFC path using Q-learning
http://<host IP running this module>:<port number>/path_selection/q_learning

# Choose and create an SFC path randomly
http://<host IP running this module>:<port number>/path_selection/random
```

Required data to create SFC is defined in SFCInfo model that is JSON format data.
The SFCInfo model consits of 4 data as follows.

- sfc_name: a name of SFC identified by OpenStack
- sfc_prefix: a prefix to identify instances which can be components of an SFC from OpenStack
- sfc_vnfs: a string array including a flow classifier name and name of each VNF instance in order
- sfcr_name: a name of flow classifier identified by OpenStack

For example, if an SFC request includes SFCInfo data as follows, this module identifies an instance of which name is test-client as a flow classifier and VNF instances of which name starts with test-firewall and test-dpi to create an SFC.

```
    {
      “sfc_name”: "sample-sfc",
      “sfc_prefix”: “test-”,
      “sfc_vnfs”: [
        “client”, “firewall”, “dpi”
      ],
      “sfcr_name”: “sample-sfcr”
    }
```
