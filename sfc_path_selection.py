from __future__ import print_function
import ni_nfvo_client
import ni_mon_client
from datetime import datetime, timedelta
from ni_nfvo_client.rest import ApiException as NfvoApiException
from ni_mon_client.rest import ApiException as NimonApiException
from config import cfg
from server.models.sfc_info import SFCInfo

import random
import numpy as np
import datetime as dt


# Parameters
# OpenStack Parameters
openstack_network_id = "" # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (Q-learning in this codes)
learning_rate = 0.10         # Learning rate
discount_factor = 0.60       # Discount factor
initial_epsilon = 0.90       # epsilon value of -greedy algorithm
num_episode = 3000           # Number of iteration for Q-learning


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



# get_vnf_info(sfc_prefix, sfc_vnfs): get each VNF instance ID and information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: Dict. object = {'vnfi_info': vnfi information, 'num_vnf_type': number of each vnf type}
def get_vnf_info(sfc_prefix, sfc_vnfs):

    # Get information of VNF instances which are used for SFC
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instances()

    selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfc_prefix + vnf_type) ]
    node_ids = [ vnfi.node_id for vnfi in selected_vnfi ]
    node_ids = list(set(node_ids))

    vnfi_list = []
    num_vnf_type = []
    temp = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        temp.append([])

        temp[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfc_prefix + vnf_type) ]
        temp[i].sort(key=lambda vnfi: vnfi.name)

        for vnfi in temp[i]:
            vnfi.node_id = node_ids.index(vnfi.node_id)

        vnfi_list = vnfi_list + temp[i]
        num_vnf_type.append(len(temp[i]))

    return {'vnfi_list': vnfi_list, 'num_vnf_type': num_vnf_type}



# get_vnf_resources(vnfi_list): get resources info. of VNF instance from the monitoring module
# Input: VNF instance list
# Output: Resource array -> [(CPU utilization, Memory utilization, Physical location), (...), ...]
def get_vnf_resources(vnfi_list):

    # In this codes, we regard CPU utilization, Memory utilization, Physicil node location
    resource_type = ("cpu_usage___value___gauge", "memory_free___value___gauge")
    ni_mon_api = get_monitoring_api()

    # Create an initial resource table initialized by 0
    resources = np.zeros((len(vnfi_list), len(resource_type)+1))

    # Query to get resource data
    for vnfi in vnfi_list:
        i = vnfi_list.index(vnfi)

        for type in resource_type:
            j = resource_type.index(type)

            vnf_id = vnfi.id
            measurement_type = type
            end_time = dt.datetime.now()
            start_time = end_time - dt.timedelta(seconds = 10)

            response = ni_mon_api.get_measurement(vnf_id, measurement_type, start_time, end_time)
            resources[i, j] = response[-1].measurement_value

            # Calculate CPU utilization as persent
            if j == 0:
                resources[i, j] = resources[i, j]

            # Calculate Memory utilization as percent
            elif j == 1:
                flavor_id = vnfi_list[i].flavor_id
                memory_query = ni_mon_api.get_vnf_flavor(flavor_id)
                memory_total = 1000000 * memory_query.ram_mb
                resources[i, j] = (resources[i, j]/memory_total)*100

        # Additionally, insert vnf location
        resources[i, -1] = vnfi.node_id

    return resources



# get_vnf_type(current_state, num_vnf_type): get vnf type showing vnf order of SFC
# Input: current state number, number of each vnf instance
# Output: vnf type (the order which is index number of vnf in sfc)
def get_vnf_type(current_state, num_vnf_type):

    index = len(num_vnf_type)
    pointer = num_vnf_type[0]

    for i in range (0, index):
        if current_state < pointer:
            return i
        else:
            pointer = pointer + num_vnf_type[i+1]



# get_action(current_state, Q, epsilon, pi_0): decide action from current state
# Input: current state, Q-value, epsilon for -greedy, action policy
# Output: action from current state
def get_action(current_state, Q, epsilon, pi_0):

    [state, action] = Q.shape

    # Decide action
    # Choose random action with probability
    if np.random.rand() < epsilon:
        next_action = np.random.choice(action, p=pi_0[current_state, :])
    # Choose the action maximizing Q-value
    else:
        next_action = np.nanargmax(Q[current_state, :])

    return next_action



# get_next_state(current_state, current_action, num_vnf_type): move from current state to next state after doing action
# Input: currrent_state, current_action, num_vnf_type
# Output: next_state (if negative value, it means no next state)
def get_next_state(current_state, current_action, num_vnf_type):

    current_vnf_type = get_vnf_type(current_state, num_vnf_type)
    last_vnf_type = len(num_vnf_type) - 1

    next_state = 0

    for i in range (0, current_vnf_type+1):
        next_state = next_state + num_vnf_type[i]

    next_state = next_state + current_action

    return next_state



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
def set_sfc(sfcr_id, sfc_name, sfc_path, vnfi_info):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()

    vnf_instance_ids= []

    for vnfi in vnfi_info:
        for vnf in sfc_path:
            if sfc_path.index(vnf) == 0:
                continue

            if vnfi.name == vnf:
                vnf_instance_ids.append([ vnfi.id ])

    sfc_spec = ni_nfvo_client.SfcSpec(sfc_name=sfc_name,
                                   sfcr_ids=[ sfcr_id ],
                                   vnf_instance_ids=vnf_instance_ids)

    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)

    return api_response



# set_action_policy(theta): define initial action policy
# Input: (number of states, number of actions, number of each vnf instance)
# Output: initial action policy
def set_action_policy(theta):

    [m, n] = theta.shape
    pi = np.zeros((m, n))

    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)  # Change nan to 0

    return pi



# set_initial_policy(vnfi_info): create initial policy array
# Input: vnfi_info
# Output: initial policy array
def set_initial_policy(vnfi_info):

    # Number of each VNF type
    # Count states and actions for RL (exclude final states)
    num_vnf_type = vnfi_info["num_vnf_type"]
    num_state = len(vnfi_info["vnfi_list"]) - num_vnf_type[-1]
    num_action = max(num_vnf_type)

    policy = np.zeros((num_state, num_action))

    final_type = len(num_vnf_type)-1

    for i in range (0, num_state):
        nan_list = []

        vnf_type = get_vnf_type(i, num_vnf_type)

        # Is it final states?
        if vnf_type == final_type:
            for j in range(0, num_action):
                nan_list.append(np.nan)

        else:
            for j in range(0, num_action):
                if j < num_vnf_type[vnf_type+1]:
                    nan_list.append(1)
                else:
                    nan_list.append(np.nan)

        policy[i] = nan_list

    return policy



# is_final_state(state, num_vnf_type): check whether input state is final state or not
# Input: currrent_state, num_vnf_type
# Output: true or false
def is_final_state(state, num_vnf_type):

    vnf_type = get_vnf_type(state, num_vnf_type)
    last_vnf_type = len(num_vnf_type) - 1

    if vnf_type == last_vnf_type:
        return True
    else:
        return False



# sample_object_creation(): create SFCInfo object to be used as a input of q_based_sfc and random_sfc
# Input: sfc_prefix, sfc_vnfs
# Output: sfc_info
def sample_object_creation(sfc_prefix, sfc_vnfs):

    sfcr_name = "sample_sfcr"
    sfc_prefix = sfc_prefix
    sfc_vnfs = sfc_vnfs
    sfc_name = "sample_sfc"

    sfc_info = SFCInfo(sfcr_name, sfc_prefix, sfc_vnfs, sfc_name)

    return sfc_info



# Q_learning(current_state, current_action, r, next_state, Q, eta, gamma): Q-leraning algorithm to updeate Q-value
# Input: current_state, current_action, rewords, next_state, Q-value, Discount factor
# Output: updated Q-value
def Q_learning(current_state, current_action, r, next_state, Q, eta, gamma):

    if next_state == -1: # Arriving final state
        Q[current_state, current_action] = Q[current_state, current_action] + eta * (r)
    else:
        Q[current_state, current_action] = Q[current_state, current_action] + eta * (r + gamma * np.nanmax(Q[next_state,: ]) - Q[current_state, current_action])

    return Q



# sfc_path_selection(Q, epsilon, eta, gamma, pi, resources, vnfi_list, num_vnf_type): decide sfc path by Q-learning
# Input: Q-value, epsilon of -greedy , learning rate, discount factor, action policy, resources, vnf instance info, number of each vnf type
# Output: history, Q-value
def sfc_path_selection(Q, epsilon, eta, gamma, pi, resources, vnfi_list, num_vnf_type):

    current_state = 0  # Starting state
    s_a_history = [[0, np.nan]]  # Define list to track history of (agent action, state)

    while (1):  # Unitl deciding SFC path
        current_action = get_action(current_state, Q, epsilon, pi)  # Choosing an action

        s_a_history[-1][1] = current_action  # Adding current state (-1 because no entry in the list now)

        next_state = get_next_state(current_state, current_action, num_vnf_type) # Get next state

        # Reward calculatoin
        ## CPU Utilization
        if resources[next_state, 0] < 1:
            r_cpu = 1
        else:
            r_cpu = 1/(100 * resources[next_state, 0])

        ## Memory Utilization
        if resources[next_state, 1] < 1:
            r_memory = 1
        else:
            r_memory = resources[next_state, 1]/100

        ## VNF location
        if resources[current_state, 2] == resources[next_state, 2]: # if exist in the same physical node
            r_location = 1
        else:
            r_location = 0

        ## Give different weights whether it is CPU intensive of Memory intensive
        vnf_type = get_vnf_type(next_state, num_vnf_type)
        vnfi_name = vnfi_list[next_state].name

        if "firewall" in vnfi_name or "flowmonitor" in vnfi_name or "proxy" in vnfi_name:
            r = (0.35*r_cpu) + (0.15*r_memory) + (0.5*r_location) ## CPU intensive (weights: CPU 0.35, Memory 0.15, Location 0.5)
        elif "dpi" in vnfi_name or "ids" in vnfi_name:
            r = (0.15*r_cpu) + (0.35*r_memory) + (0.5*r_location) ## Memory intensive (weights: CPU 0.15, Memory 0.35, location 0.5)
        else:
            r = (0.40*r_cpu) + (0.30*r_memory) + (0.30*r_location) ## Others
            
        s_a_history.append([next_state, np.nan]) # Adding next state into the history

        # Update Q-value
        if is_final_state(next_state, num_vnf_type):
            Q = Q_learning(current_state, current_action, r, -1, Q, eta, gamma)
        else:
            Q = Q_learning(current_state, current_action, r, next_state, Q, eta, gamma)

        # Check wheter final state or not
        if is_final_state(next_state, num_vnf_type):
            break
        else:
            current_state = next_state

    return [s_a_history, Q]



# q_based_sfc(sfcr_name, sfc_vnfs, sfc_name): create sfc by using Q-leraning
# Input: JSON sfc_info (flowclassifier name, sfc vnfs, sfc name)
# Output: flow classifier id, sfc id
def q_based_sfc(sfc_info):

    eta = learning_rate       # Learning rate
    gamma = discount_factor     # Discount factor
    epsilon = initial_epsilon   # epsilon value of -greedy algorithm
    episode = num_episode   # Number of iteration for Q-learning

    ## Step 1: Get VNF instance Info
    vnfi_info = get_vnf_info(sfc_info.sfc_prefix, sfc_info.sfc_vnfs)

    vnfi_list = vnfi_info["vnfi_list"]
    num_vnf_type = vnfi_info["num_vnf_type"]

    vnf_resources = get_vnf_resources(vnfi_list)

    ## Step 2: initialize Q-value and action policy
    pi_0 = set_initial_policy(vnfi_info)
    pi_0 = set_action_policy(pi_0)
    Q = pi_0 * 0

    ## Step 3: Q-Leraning
    for i in range(0, episode):

        # Decrese epsilon value
        epsilon = 1./((i / 100) + 1)

        # Finding SFC path
        [s_a_history, Q] = sfc_path_selection(Q, epsilon, eta, gamma, pi_0, vnf_resources, vnfi_list, num_vnf_type)

    ## Step 4: Print final sfc path decision regarding to the latest Q-value
    sfc_path = []
    epsilon = 0

    [s_a_history, Q] = sfc_path_selection(Q, epsilon, eta, gamma, pi_0, vnf_resources, vnfi_list, num_vnf_type)

    for i in range(0, len(s_a_history)):
        sfc_path.append(vnfi_list[s_a_history[i][0]].name)

    print(sfc_path)

    ## Step 5: Create sfc in the real environment
    # create flow classifier
    # Get source client VM ID
    for vnfi in vnfi_list:
        if sfc_path[0] == vnfi.name:
            source_client = vnfi.id
            break

    # Extract ip prefix of a flow classifier
    src_ip_prefix = get_ip_from_id(source_client) + "/32"

    sfcr_id = set_flow_classifier(sfc_info.sfcr_name, src_ip_prefix, sfc_info.sfc_vnfs, source_client)
    sfc_id = set_sfc(sfcr_id, sfc_info.sfc_name, sfc_path, vnfi_list)

    response = { "sfcr_id": sfcr_id,
                 "sfc_id": sfc_id,
                 "sfc_path": sfc_path }

    return response


# random_sfc(sfcr_name, sfc_vnfs, sfc_name):
# Input: JSON sfc_info (flowclassifier name, sfc vnfs, sfc name)
# Output: flow classifier id, sfc id
def random_sfc(sfc_info):

    ## Step 1: Get VNF instance Info
    vnfi_info = get_vnf_info(sfc_info.sfc_prefix, sfc_info.sfc_vnfs)

    vnfi_list = vnfi_info["vnfi_list"]
    num_vnf_type = vnfi_info["num_vnf_type"]

    # Random sfc path selection
    start = 0
    end = 0
    sfc_path = []

    for i in range(0, len(num_vnf_type)):
        end = start + num_vnf_type[i]
        sfc_path.append(random.choice(vnfi_list[start:end]).name)
        start = start + num_vnf_type[i]

    print(sfc_path)

    # Get flow classifier's id
    for vnfi in vnfi_list:
        if vnfi.name == sfc_path[0]:
            source_client = vnfi.id
            break

    # Extract ip prefix of a flow classifier
    src_ip_prefix = get_ip_from_id(source_client) + "/32"

    sfcr_id = set_flow_classifier(sfc_info.sfcr_name, src_ip_prefix, sfc_info.sfc_vnfs, source_client)
    sfc_id = set_sfc(sfcr_id, sfc_info.sfc_name, sfc_path, vnfi_list)

    response = { "sfcr_id": sfcr_id,
                 "sfc_id": sfc_id,
                 "sfc_path": sfc_path }

    return response
