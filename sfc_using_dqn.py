import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from datetime import datetime, timedelta
from config import cfg
from torch_dqn import *

import numpy as np

import datetime as dt
import math
import os
import time
import subprocess
from pprint import pprint
import random
import json

# Parameters
# OpenStack Parameters
openstack_network_id = "9cdee37a-fdcd-45f5-860b-253fc62ce578" # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 10000           # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons = 64               # Number of neurons in each hidden layer
epsilon = 0.99                  # epsilon value of e-greedy algorithm
required_mem_size = 24        # Minimum number triggering sampling
print_interval = 24             # Number of iteration to print result during DQN

# Global values
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"
sfc_update_flag = True
training_list = []

# get_monitoring_api(): get ni_monitoring_client api to interact with a monitoring module
# Input: null
# Output: monitoring moudle's client api
def get_monitoring_api():

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    return ni_mon_api


# get_nfvo_sfc_api(): get ni_nfvo_sfc api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfc api
def get_nfvo_sfc_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfc_api


# get_nfvo_sfcr_api(): get ni_nfvo_sfcr api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfcr api
def get_nfvo_sfcr_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfcr_api


# get_nfvo_vnf_api(): get ni_nfvo_vnf api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf api
def get_nfvo_vnf_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_vnf_api


# get_all_flavors(): get all flavors information
# Input: null
# Output: flavors information
def get_all_flavors():

    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_flavors()

    return query


# destroy_vnf(id): destory VNF instance in OpenStack environment
# Inpurt: ID of VNF instance
# Output: API response
def destroy_vnf(id):

    ni_nfvo_api = get_nfvo_vnf_api()
    api_response = ni_nfvo_api.destroy_vnf(id)

    return api_response


# get_vnf_info(sfc_prefix, sfc_vnfs): get each VNF instance information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple or list [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: VNF information list
def get_vnf_info(sfc_prefix, sfc_vnfs):

    # Get information of VNF instances which are used for SFC
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instances()

    selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfc_prefix + vnf_type) ]

    vnfi_list = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        vnfi_list.append([])

        vnfi_list[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfc_prefix + vnf_type) ]
        vnfi_list[i].sort(key=lambda vnfi: vnfi.name)

    return vnfi_list


# get_specific_vnf_info(sfc_prefix, id): get specific VNF instance information from monitoring module
# Input: VNF instance ID
# Output: VNF information
def get_specific_vnf_info(id):
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instance(id)

    return query


# set_flow_classifier(sfcr_name, sfc_ip_prefix, nf_chain, source_client): create flow classifier in the testbed
# Input: flowclassifier name, flowclassifier ip prefix, list[list[each vnf id]], flowclassifier VM ID
# Output: response
def set_flow_classifier(sfcr_name, src_ip_prefix, nf_chain, source_client):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()

    sfcr_spec = ni_nfvo_client.SfcrSpec(name=sfcr_name,
                                 src_ip_prefix=src_ip_prefix,
                                 nf_chain=nf_chain,
                                 source_client=source_client)

    api_response = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)

    return api_response


# set_sfc(sfcr_id, sfc_name, sfc_path, vnfi_list): create sfc in the testbed
# Input: flowclassifier name, sfc name, sfc path, vnfi_info
# Output: response
def set_sfc(sfcr_id, sfc_name, inst_in_sfc):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()

    del inst_in_sfc[0]
    instIDs = []

    for inst in inst_in_sfc:
        instIDs.append([ inst.id ])

    sfc_spec = ni_nfvo_client.SfcSpec(sfc_name=sfc_name,
                                   sfcr_ids=[ sfcr_id ],
                                   vnf_instance_ids=instIDs)

    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)

    return api_response


# get_instance_info(instance, flavor): create sfc in the testbed
# Input: flowclassifier name, sfc name, sfc path, vnfi_info
# Output: response
def get_instance_info(instance, flavor):
    ni_mon_api = get_monitoring_api()
    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge"]

    info = { "id": instance.id, "cpu" : 0.0, "memory": 0.0}

    # Set time-period to get resources
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(seconds = 10)

    for resource in resource_type:
        query = ni_mon_api.get_measurement(instance.id, resource, start_time, end_time)
        value = 0

        for response in query:
            value = value + response.measurement_value

        value = value/len(query) if len(query) > 0 else 0

        if resource.startswith("cpu"):
            info["cpu"] = value
        elif resource.startswith("memory"):
            memory_ram_mb = flavor.ram_mb
            memory_total = 1000000 * memory_ram_mb
            info["memory"] = 100*(1-(value/memory_total)) if len(query) > 0 else 0

    return info


# get_ip_from_vm(vm_id):
# Input: vm instance id
# Output: port IP of the data plane
def get_ip_from_id(vm_id):

    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instance(vm_id)

    ## Get ip address of specific network
    ports = query.ports
    network_id = openstack_network_id

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]



def get_hops_in_topology(src_node, dst_node):

    nodes = [ "NI-Compute-82-81", "NI-Compute-82-82", "ni-compute-82-48", "NI-Compute-82-49", "NI-Compute-82-51", "NI-Compute-82-55"]
    hops = [[1, 2, 4, 4, 4, 5],
            [2, 1, 4, 4, 4, 5],
            [4, 4, 1, 2, 2, 5],
            [4, 4, 2, 1, 2, 5],
            [4, 4, 2, 2, 1, 5],
            [5, 5, 5, 5, 5, 1]]

    return hops[nodes.index(src_node)][nodes.index(dst_node)]

# get_node_info(): get all node information placed in environment
# Input: null
# Output: Node information list
def get_node_info(flavor):
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_nodes()

    response = [ node_info for node_info in query if node_info.type == "compute" and node_info.status == "enabled"]
    response = [ node_info for node_info in response if not (node_info.name).startswith("NI-Compute-82-9")]
    response = [ node_info for node_info in response if node_info.n_cores_free >= flavor.n_cores and node_info.ram_mb >= flavor.ram_mb]

    return response


# get_nfvo_vnf_spec(): get ni_nfvo_vnf spec to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf spec
def get_nfvo_vnf_spec():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(ni_nfvo_client.ApiClient(nfvo_client_cfg))
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec

# deploy_vnf(vnf_spec): deploy VNF instance in OpenStack environment
# Input: VnFSpec defined in nfvo client module
# Output: API response
def deploy_vnf(vnf_spec):

    ni_nfvo_api = get_nfvo_vnf_api()
    instID = ni_nfvo_api.deploy_vnf(vnf_spec)

    for i in range (0, 30):
        time.sleep(2)

        if check_active_instance(instID):
            return get_specific_vnf_info(instID)

    return ""


# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
    api = get_monitoring_api()
    status = api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


def reward_calculator(src, dst):
    cost = 1.25
    resTime = 0

    for port in src.ports:
        if port.network_id == openstack_network_id:
            src_ip = port.ip_addresses[-1]
            break

    for port in dst.ports:
        if port.network_id == openstack_network_id:
            dst_ip = port.ip_addresses[-1]
            break

    for i in range (0, 15):
        time.sleep(2)

        command = ("sshpass -p %s ssh %s@%s ./test_ping_e2e.sh %s %s %s %s" % (cfg["instance"]["password"],
                                                                               cfg["instance"]["id"],
                                                                               cfg["instance"]["monitor"],
                                                                               src_ip,
                                                                               cfg["instance"]["id"],
                                                                               cfg["instance"]["password"],
                                                                               dst_ip))
        command = command + " | grep avg | awk '{split($4,a,\"/\");print a[2]}'"

        resTime = subprocess.check_output(command, shell=True).strip().decode("utf-8")

        if resTime != "":
            resTime = float(resTime)/1000.0
            reward = -math.log(1.0+resTime)*cost

            return reward


    return 10


def dqn_training(sfc_info):
    epsilon_value = epsilon
    n_epi = 0
    training_list.append(sfc_info.sfc_name)

    # Q-network, Target Q-network, remplay memory
    q = Qnet(3, 2, 32) # State 3, Action 2, Neuron 32
    q_target = Qnet(3, 2, 32) # State 3, Action 2, Neuron 32
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    memory = ReplayBuffer(buffer_limit)

    flavor_info = get_all_flavors()

    while True:
        time.sleep(10)
        vnf_info = get_vnf_info(sfc_info.sfc_prefix, sfc_info.sfc_vnfs)

        # Insert Traffic classifer instance
        deployedInst = []
        instInSFC = []
        instInSFC.append(vnf_info[0][-1])
        del vnf_info[0]

        # Create State (Except for Traffic classifer)
        for vnf in vnf_info:
            cpuUtil = 0
            memUtil = 0
            placement = 0
            instSize = len(vnf)
            resourceInfo = []

            # Measure mean values per each VNF type
            for inst in vnf:
                flavor = [ flavor for flavor in flavor_info if flavor.id == inst.flavor_id ][-1]
                inst_resUtil = get_instance_info(inst, flavor)

                cpuUtil = cpuUtil + (inst_resUtil["cpu"]/instSize)
                memUtil = memUtil + (inst_resUtil["memory"]/instSize)
                placement = placement + get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)/instSize
                resourceInfo.append({"id": inst.id, "cpu": inst_resUtil["cpu"], "memory": inst_resUtil["memory"], "placement": get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)})

            # Create state
            state = np.array([cpuUtil, memUtil, placement])
            epsilon_value = max(0.10, epsilon_value*0.99)
            action = q.sample_action(torch.from_numpy(state).float(), epsilon_value)["action"]


            if action == 0: # Select an instance
                resourceInfo.sort(key=lambda info: info["placement"])
                resourceInfo.sort(key=lambda info: info["cpu"])
                instInSFC.append([ inst for inst in vnf if inst.id == resourceInfo[0]["id"] ][-1])

            elif action == 1: # Deploy a new instance
                node_info = get_node_info(flavor)
                node_info = [ {"id": node.id, "distance": get_hops_in_topology(instInSFC[-1].node_id, node.id) } for node in node_info if node.id != "NI-Compute-Handong"]
                node_info.sort(key=lambda info: info["distance"])

                vnf_spec = get_nfvo_vnf_spec()
                vnf_type = sfc_info.sfc_vnfs[vnf_info.index(vnf)+1]
                vnf_spec.vnf_name = sfc_info.sfc_prefix + vnf_type + " " + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                vnf_spec.image_id = cfg["image"][vnf_type]
                vnf_spec.flavor_id = flavor.id
                vnf_spec.node_name = node_info[0]["id"]

                # After successful deployment, add information of the deployed instance
                newInst = deploy_vnf(vnf_spec)

                if newInst != "":
                    deployedInst.append(newInst)
                    instInSFC.append(newInst)
                else:
                    instInSFC.append(random.choice(vnf))


            # Reward calculation
            time.sleep(10)
            length = len(instInSFC)

            reward = reward_calculator(instInSFC[length-2], instInSFC[length-1])

            # Create new state
            flavor = [ flavor for flavor in flavor_info if flavor.id == instInSFC[-1].flavor_id ][-1]
            inst_resUtil = get_instance_info(instInSFC[-1], flavor)
            new_cpuUtil = ((cpuUtil*instSize)+inst_resUtil["cpu"])/(instSize+1)
            new_memUtil = ((memUtil*instSize)+inst_resUtil["memory"])/(instSize+1)
            new_placement = ((placement*instSize)+get_hops_in_topology(instInSFC[-2].node_id, instInSFC[-1].node_id))/(instSize+1)
            nextState = np.array([new_cpuUtil, new_memUtil, new_placement])

            # Store in Replay memory
            transition = (state, action, reward, nextState, 1.0)
            memory.put(transition)

            if memory.size() > required_mem_size:
                train(q, q_target, memory, optimizer, gamma, batch_size)

            if n_epi % print_interval==0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())

            n_epi = n_epi+1

            if len(instInSFC) == len(sfc_info.sfc_vnfs):
                for inst in deployedInst:
                    destroy_vnf(inst.id)

            # Finish
            if sfc_info.sfc_name not in training_list:
                q.save_model("./dqn_models/"+sfc_info.sfc_name)

                for inst in deployedInst:
                    destroy_vnf(inst.id)

                print("[Training finish] " + sfc_info.sfc_name)


def dqn_based_sfc(sfc_info):

    q = Qnet(3, 2, 32) # State 3, Action 2, Neuron 32
    q.load_state_dict(torch.load("./dqn_models/" + sfc_info.sfc_name))

    flavor_info = get_all_flavors()
    vnf_info = get_vnf_info(sfc_info.sfc_prefix, sfc_info.sfc_vnfs)

    # Insert Traffic classifer instance
    instInSFC = []
    instInSFC.append(vnf_info[0][-1])
    del vnf_info[0]

    # Create state
    for vnf in vnf_info:
        cpuUtil = 0
        memUtil = 0
        placement = 0
        instSize = len(vnf)
        resourceInfo = []

        # Measure mean values
        for inst in vnf:
            flavor = [ flavor for flavor in flavor_info if flavor.id == inst.flavor_id ][-1]
            inst_resUtil = get_instance_info(inst, flavor)

            cpuUtil = cpuUtil + (inst_resUtil["cpu"]/instSize)
            memUtil = memUtil + (inst_resUtil["memory"]/instSize)
            placement = placement + get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)/instSize
            resourceInfo.append({"id": inst.id, "cpu": inst_resUtil["cpu"], "memory": inst_resUtil["memory"], "placement": get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)})

        # Create state
        state = np.array([cpuUtil, memUtil, placement])
        action = q.sample_action(torch.from_numpy(state).float(), 0)["action"]


        if action == 0: # Select
            resourceInfo.sort(key=lambda info: info["placement"])
            resourceInfo.sort(key=lambda info: info["cpu"])
            instInSFC.append([ inst for inst in vnf if inst.id == resourceInfo[0]["id"] ][-1])

        elif action == 1: # Deploy
            node_info = get_node_info(flavor)
            node_info = [ {"id": node.id, "distance": get_hops_in_topology(instInSFC[-1].node_id, node.id) } for node in node_info if node.id != "NI-Compute-Handong"]
            node_info.sort(key=lambda info: info["distance"])

            vnf_spec = get_nfvo_vnf_spec()
            vnf_type = sfc_info.sfc_vnfs[vnf_info.index(vnf)+1]
            vnf_spec.vnf_name = sfc_info.sfc_prefix + vnf_type + " " + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            vnf_spec.image_id = cfg["image"][vnf_type]
            vnf_spec.flavor_id = flavor.id
            vnf_spec.node_name = node_info[0]["id"]

            # Deployment success
            newInst = deploy_vnf(vnf_spec)

            if newInst != "":
                instInSFC.append(newInst)
            else:
                instInSFC.append(random.choice(vnf))


    # Create SFC
    ipPrefix = get_ip_from_id(instInSFC[0].id) + "/32"
    sfcrID = set_flow_classifier(sfc_info.sfcr_name, ipPrefix, sfc_info.sfc_vnfs, instInSFC[0].id)
    sfcID = set_sfc(sfcrID, sfc_info.sfc_name, instInSFC)
    sfcPath = [ inst.name for inst in instInSFC ]

    response = { "sfcr_id": sfcrID,
                 "sfc_id": sfcID,
                 "sfc_path": sfcPath }

    pprint(sfcPath)

    return response
