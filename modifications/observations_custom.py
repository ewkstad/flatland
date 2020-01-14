"""
Collection of environment-specific ObservationBuilder.
"""
import collections
from typing import Optional, List, Dict, Tuple
from copy import deepcopy
from scipy.ndimage import rotate
from scipy.sparse.csr import csr_matrix

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet

RAD = 5

feat_order = {'own_target_encountered':0,
              'other_targets_encountered':1,
              'agent_rel_direction_F':2,
              'agent_rel_direction_R':3,
              'agent_rel_direction_B':4,
              'agent_rel_direction_L':5,
              'potential_conflict':6,
              'unusable_switch':7,
              'dist_min_to_target':8,
              'agent_malfunctioning':9,
              'speed_min_fractional':10,
              'other_agent_slower':11,
              'other_agent_same_speed':12,
              'other_agent_faster':13,
              'agent_ready_to_depart':14,
              'path_rel_direction_F':15,
              'path_rel_direction_R':16,
              'path_rel_direction_B':17,
              'path_rel_direction_L':18,
              'abs_y_pos_env':19,
              'abs_x_pos_env':20,
              'abs_y_inv_dist_kernel':21,
              'abs_x_inv_dist_kernel':22,
              'agent_active':23,
              'inv_dist_min_to_target':24,
              'max_diff':None,
              }

feat_vec_max = len(feat_order) - 1

rel_direction = {0:'F',
                 1:'R',
                 2:'B',
                 3:'L'}


def rotate_map(maps, dir_):
    rot_deg = (dir_-4)*90
    if rot_deg == -360: rot_deg = 0
    if rot_deg != 0:
        maps = rotate(maps, rot_deg)
    return maps

def get_rotation_dict():
    xx, yy = np.meshgrid(np.arange(2*RAD+1), np.arange(2*RAD+1))
    idx_mat = yy*(2*RAD+1) + xx
    rotation_dict = dict()
    for dir_ in range(0, 4):
        rotation_dict[dir_] = rotate_map(idx_mat, dir_).ravel()
    return rotation_dict

rotation_dict = get_rotation_dict()

class TreeObsForRailEnvAug(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """
    Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                                          'dist_other_target_encountered '
                                          'dist_other_agent_encountered '
                                          'dist_potential_conflict '
                                          'dist_unusable_switch '
                                          'dist_to_next_branch '
                                          'dist_min_to_target '
                                          'num_agents_same_direction '
                                          'num_agents_opposite_direction '
                                          'num_agents_malfunctioning '
                                          'speed_min_fractional '
                                          'num_agents_ready_to_depart '
                                          'childs')

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int, RAD: int, predictor: PredictionBuilder = None):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 11
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = None
        self.search_obs = {}
        self.search_branches = {}
        self.sub_targets_graph = None
        self.neighbors = {}
        self.active_branch = None
        self.break_search = False
        self.root_pos = None
        self.RAD = RAD
        self.digraphs = {}

    def reset(self):
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}

    def get_active_branch(self):
        return self.active_branch

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> Node:
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        self.search_obs[handle] = None
        self.search_branches[handle] = None
        self.neighbors[handle] = []
        self.digraphs[handle] = None

        digraph = csr_matrix(((2*self.RAD + 1)**2, (2*self.RAD + 1)**2))

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        agent_ready_to_depart = 0
        agent_active = 0
        own_target_encountered = 0

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
            agent_ready_to_depart = 1
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
            agent_active = 1
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
            own_target_encountered = 1
        else:
            return None

        self.search_obs[handle] = dict()
        self.search_branches[handle] = dict()
        self.neighbors[handle] = []

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()
        dist_min_to_target = distance_map[(handle, *agent_virtual_position, agent.direction)]
        agent_malfunctioning = agent.malfunction_data['malfunction']
        speed_min_fractional = agent.speed_data['speed']
        dist_scalar = 0
        if dist_min_to_target != np.inf:
            dist_scalar = dist_min_to_target

        root_node_observation = TreeObsForRailEnvAug.Node(dist_own_target_encountered=0,
                                                          dist_other_target_encountered=0,
                                                          dist_other_agent_encountered=0,
                                                          dist_potential_conflict=0,
                                                          dist_unusable_switch=0,
                                                          dist_to_next_branch=0,
                                                          dist_min_to_target=dist_min_to_target,
                                                          num_agents_same_direction=0,
                                                          num_agents_opposite_direction=0,
                                                          num_agents_malfunctioning=agent_malfunctioning,
                                                          speed_min_fractional=speed_min_fractional,
                                                          num_agents_ready_to_depart=0,
                                                          childs={})

        visited = OrderedSet()
        visited_features = dict()
        branches = dict()

        featvec = [None] * len(feat_order)
        featvec[feat_order['own_target_encountered']] = own_target_encountered
        featvec[feat_order['agent_ready_to_depart']] = agent_ready_to_depart
        featvec[feat_order['agent_active']] = agent_active
        featvec[feat_order['dist_min_to_target']] = dist_min_to_target
        featvec[feat_order['inv_dist_min_to_target']] = dist_min_to_target
        featvec[feat_order['agent_malfunctioning']] = agent_malfunctioning
        featvec[feat_order['speed_min_fractional']] = speed_min_fractional
        featvec[feat_order['abs_y_pos_env']] = agent_virtual_position[0] / self.env.height
        featvec[feat_order['abs_x_pos_env']] = agent_virtual_position[1] / self.env.width
        featvec[feat_order['abs_y_inv_dist_kernel']] = 1
        featvec[feat_order['abs_x_inv_dist_kernel']] = 1

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        self.root_pos = (agent_virtual_position[0], agent_virtual_position[1])

        self.break_search = False

        root_idx = self.RAD*(2*self.RAD+1) + self.RAD

        digraph[root_idx, root_idx] = 1

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:

                featvec[feat_order['path_rel_direction_{}'.format(rel_direction[(branch_direction-orientation)%4])]] = 1

                new_cell = get_new_position(agent_virtual_position, branch_direction)

                self.active_branch = '{}_{}_{}_{}'.format(0, self.root_pos[0], self.root_pos[1], branch_direction)

                branches[self.active_branch] = [(self.root_pos, tuple(new_cell[:2]))]

                branch_observation, branch_visited, visited_features, dist_scalar, branches, digraph = self._explore_branch(handle,
                                                                                                                            self.root_pos,
                                                                                                                            new_cell,
                                                                                                                            orientation,
                                                                                                                            branch_direction,
                                                                                                                            visited_features,
                                                                                                                            1,
                                                                                                                            1,
                                                                                                                            dist_scalar,
                                                                                                                            branches,
                                                                                                                            digraph)
                root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited
        self.search_obs[handle] = visited_features
        self.search_obs[handle]['{}_{}_{}_{}'.format(0, agent_virtual_position[0], agent_virtual_position[1], orientation)] = featvec
        self.search_obs[handle]['distance_scalar'] = dist_scalar
        self.search_branches[handle] = branches
        digraph = digraph[:, rotation_dict[agent.direction]]
        digraph = digraph[rotation_dict[agent.direction]]
        self.digraphs[handle] = digraph
        return root_node_observation


    def _explore_branch(self, handle, root_position, position, initial_direction, direction, visited_features, tot_dist, depth, distance_scalar, branches, digraph):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], [], visited_features, distance_scalar, branches, digraph

        active_branch = self.get_active_branch()

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        speed = agent.speed_data['speed']
        time_per_cell = np.reciprocal(speed)
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0
        dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        root_idx = (2*self.RAD+1)*(root_position[0]-self.root_pos[0]+self.RAD) + (root_position[1]-self.root_pos[1]+self.RAD)
        p_pos = (position[0], position[1])
        y, x = (position[0] - self.root_pos[0] + self.RAD), (position[1] - self.root_pos[1] + self.RAD)
        p_idx = (2*self.RAD+1)*y + x
        fully_connected_sub_branch = []
        s_idx = None

        # Root and Successor are feasible
        if not self.break_search and ((y > -1) and y < (2 * self.RAD + 1) and (x > -1) and (x < 2 * self.RAD + 1)) and dist_min_to_target != np.inf:
            digraph[root_idx, p_idx] = 1

        self.break_search = False

        while exploring:

            y, x = position[0] - self.root_pos[0] + self.RAD, position[1] - self.root_pos[1] + self.RAD

            if not ((y > -1) and y < (2 * self.RAD + 1) and (x > -1) and (x < 2 * self.RAD + 1)):
                self.break_search = True
                exploring = False
                continue

            if dist_min_to_target != np.inf:
                digraph[p_idx, p_idx] = 1 # Feasible

            if s_idx is not None and dist_min_to_target != np.inf:
                digraph[p_idx, s_idx] = 1

            featvec = [None]*len(feat_order)

            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.

            featvec[feat_order['path_rel_direction_{}'.format(rel_direction[(direction-initial_direction)%4])]] = 1

            featvec[feat_order['abs_y_pos_env']] = position[0] / self.env.height
            featvec[feat_order['abs_x_pos_env']] = position[1] / self.env.width

            featvec[feat_order['abs_y_inv_dist_kernel']] = 1 - abs(position[0] - self.root_pos[0]) / (self.RAD + 1)
            featvec[feat_order['abs_x_inv_dist_kernel']] = 1 - abs(position[1] - self.root_pos[1]) / (self.RAD + 1)

            if position in self.location_has_agent:

                nghbr = self.env.agent_positions[position[0], position[1]]

                if nghbr is not -1:
                    self.neighbors[handle].append(nghbr)

                other_agent_direction = self.location_has_agent_direction[position]

                featvec[feat_order['agent_rel_direction_{}'.format(rel_direction[(other_agent_direction-initial_direction)%4])]] = 1

                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]
                    featvec[feat_order['agent_malfunctioning']] = 1

                location_has_agent_ready_to_depart = self.location_has_agent_ready_to_depart.get(position, 0)

                other_agent_ready_to_depart_encountered += location_has_agent_ready_to_depart
                featvec[feat_order['agent_ready_to_depart']] = int(location_has_agent_ready_to_depart > 0)
                featvec[feat_order['agent_active']] = 1-int(location_has_agent_ready_to_depart > 0)

                if other_agent_direction == direction:
                    # Cumulate the number of agents on branch with same direction
                    other_agent_same_direction += self.location_has_agent_direction.get((position, direction), 0)

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed
                        featvec[feat_order['speed_min_fractional']] = current_fractional_speed

                    if current_fractional_speed < speed:
                        featvec[feat_order['other_agent_slower']] = 1
                    elif current_fractional_speed == speed:
                        featvec[feat_order['other_agent_same_speed']] = 1
                    elif current_fractional_speed > speed:
                        featvec[feat_order['other_agent_faster']] = 1

                    # Other direction agents
                    # TODO: Test that this behavior is as expected
                    other_agent_opposite_direction += self.location_has_agent[position] - self.location_has_agent_direction.get((position, direction), 0)

                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    other_agent_opposite_direction += self.location_has_agent[position]

            # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist
                featvec[feat_order['unusable_switch']] = 1

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                featvec[feat_order['potential_conflict']] = 1
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                featvec[feat_order['potential_conflict']] = 1

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                                featvec[feat_order['potential_conflict']] = 1

                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                featvec[feat_order['potential_conflict']] = 1

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[post_step][ca])] == 1 and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                                featvec[feat_order['potential_conflict']] = 1

                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                                featvec[feat_order['potential_conflict']] = 1

            if position in self.location_has_target and position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist
                    featvec[feat_order['other_targets_encountered']] = 1

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist
                featvec[feat_order['own_target_encountered']] = 1

            featvec[feat_order['dist_min_to_target']] = dist_min_to_target
            featvec[feat_order['inv_dist_min_to_target']] = dist_min_to_target
            if dist_min_to_target != np.inf:
                distance_scalar = max(distance_scalar, dist_min_to_target)
            #
            # if dist_min_to_target < dist_best_to_target:
            #     sub_target_info = ((position[0], position[1]), tot_dist)
            #     dist_best_to_target = dist_min_to_target

            dist_min_to_target -= 1

            # #############################
            # #############################

            visited_features['{}_{}_{}_{}'.format(tot_dist, position[0], position[1], direction)] = featvec

            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                p_pos = (position[0], position[1]) # predecessor position
                fully_connected_sub_branch.append(p_pos)
                break

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    p_pos = (position[0], position[1]) # predecessor position
                    p_idx = (2 * self.RAD + 1) * y + x
                    position = get_new_position(position, direction)
                    s_pos = (position[0], position[1]) # successor position
                    s_idx = (2*self.RAD+1)*(s_pos[0]-self.root_pos[0]+self.RAD)+(s_pos[1]-self.root_pos[1]+self.RAD)
                    fully_connected_sub_branch.append(p_pos)
                    num_steps += 1
                    tot_dist += 1

            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0], position[1], direction)
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if self.break_search and s_idx is not None and dist_min_to_target != np.inf: # Predecessor node is feasible, but Successor node is infeasible
            digraph[p_idx, p_idx] = 1
        elif not self.break_search and s_idx is not None and dist_min_to_target != np.inf: # Successor node is terminal and feasible
            digraph[p_idx, s_idx] = 1

        if last_is_terminal or last_is_switch:
            if fully_connected_sub_branch:
                branches[active_branch].append(tuple(fully_connected_sub_branch))
                branches[active_branch].append((p_pos, s_pos))
        elif self.break_search and fully_connected_sub_branch:
            branches[active_branch].append(tuple(fully_connected_sub_branch))

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]

        if dist_min_to_target != np.inf:
            distance_scalar = max(distance_scalar, dist_min_to_target)

        node = TreeObsForRailEnvAug.Node(dist_own_target_encountered=own_target_encountered,
                                         dist_other_target_encountered=other_target_encountered,
                                         dist_other_agent_encountered=other_agent_encountered,
                                         dist_potential_conflict=potential_conflict,
                                         dist_unusable_switch=unusable_switch,
                                         dist_to_next_branch=dist_to_next_branch,
                                         dist_min_to_target=dist_min_to_target,
                                         num_agents_same_direction=other_agent_same_direction,
                                         num_agents_opposite_direction=other_agent_opposite_direction,
                                         num_agents_malfunctioning=malfunctioning_agent,
                                         speed_min_fractional=min_fractional_speed,
                                         num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                                         childs={})

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions

        root_position = (position[0], position[1])
        root_idx = (2*self.RAD+1)*(root_position[0]-self.root_pos[0]+self.RAD)+(root_position[1]-self.root_pos[1]+self.RAD)

        if not self.break_search: # Root is feasible
            digraph[root_idx, root_idx] = 1

        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back

                new_cell = get_new_position(position, (branch_direction + 2) % 4)

                self.active_branch = '{}_{}_{}_{}'.format(tot_dist, root_position[0], root_position[1], branch_direction)

                branches[self.active_branch] = [(root_position, tuple(new_cell[:2]))]

                branch_observation, branch_visited, visited_features, distance_scalar, branches, digraph = self._explore_branch(
                    handle,
                    root_position,
                    new_cell,
                    initial_direction,
                    (branch_direction + 2) % 4,
                    visited_features,
                    tot_dist + 1,
                    depth + 1,
                    distance_scalar,
                    branches,
                    digraph
                )
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            elif last_is_switch and possible_transitions[branch_direction]:

                new_cell = get_new_position(position, branch_direction)

                self.active_branch = '{}_{}_{}_{}'.format(tot_dist, root_position[0], root_position[1], branch_direction)

                branches[self.active_branch] = [(root_position, tuple(new_cell[:2]))]

                branch_observation, branch_visited, visited_features, distance_scalar, branches, digraph = self._explore_branch(
                    handle,
                    root_position,
                    new_cell,
                    initial_direction,
                    branch_direction,
                    visited_features,
                    tot_dist + 1,
                    depth + 1,
                    distance_scalar,
                    branches,
                    digraph
                )
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                node.childs[self.tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited, visited_features, distance_scalar, branches, digraph

    def util_print_obs_subtree(self, tree: Node):
        """
        Utility function to print tree observations returned by this object.
        """
        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)


class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),\
          assuming 16 bits encoding of transitions.

        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self):
        super(GlobalObsForRailEnv, self).__init__()

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 5)) - 1

        # TODO can we do this more elegantly?
        # for r in range(self.env.height):
        #     for c in range(self.env.width):
        #         obs_agents_state[(r, c)][4] = 0
        obs_agents_state[:, :, 4] = 0

        obs_agents_state[agent_virtual_position][0] = agent.direction
        obs_targets[agent.target][0] = 1

        for i in range(len(self.env.agents)):
            other_agent: EnvAgent = self.env.agents[i]

            # ignore other agents not in the grid any more
            if other_agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            obs_targets[other_agent.target][1] = 1

            # second to fourth channel only if in the grid
            if other_agent.position is not None:
                # second channel only for other agents
                if i != handle:
                    obs_agents_state[other_agent.position][1] = other_agent.direction
                obs_agents_state[other_agent.position][2] = other_agent.malfunction_data['malfunction']
                obs_agents_state[other_agent.position][3] = other_agent.speed_data['speed']
            # fifth channel: all ready to depart on this position
            if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                obs_agents_state[other_agent.initial_position][4] += 1
        return self.rail_obs, obs_agents_state, obs_targets


class LocalObsForRailEnv(ObservationBuilder):
    """
    !!!!!!WARNING!!! THIS IS DEPRACTED AND NOT UPDATED TO FLATLAND 2.0!!!!!
    Gives a local observation of the rail environment around the agent.
    The observation is composed of the following elements:

        - transition map array of the local environment around the given agent, \
          with dimensions (view_height,2*view_width+1, 16), \
          assuming 16 bits encoding of transitions.

        - Two 2D arrays (view_height,2*view_width+1, 2) containing respectively, \
        if they are in the agent's vision range, its target position, the positions of the other targets.

        - A 2D array (view_height,2*view_width+1, 4) containing the one hot encoding of directions \
          of the other agents at their position coordinates, if they are in the agent's vision range.

        - A 4 elements array with one hot encoding of the direction.

    Use the parameters view_width and view_height to define the rectangular view of the agent.
    The center parameters moves the agent along the height axis of this rectangle. If it is 0 the agent only has
    observation in front of it.

    .. deprecated:: 2.0.0
    """

    def __init__(self, view_width, view_height, center):

        super(LocalObsForRailEnv, self).__init__()
        self.view_width = view_width
        self.view_height = view_height
        self.center = center
        self.max_padding = max(self.view_width, self.view_height - self.center)

    def reset(self):
        # We build the transition map with a view_radius empty cells expansion on each side.
        # This helps to collect the local transition map view when the agent is close to a border.
        self.max_padding = max(self.view_width, self.view_height)
        self.rail_obs = np.zeros((self.env.height,
                                  self.env.width, 16))
        for i in range(self.env.height):
            for j in range(self.env.width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        agents = self.env.agents
        agent = agents[handle]

        # Correct agents position for padding
        # agent_rel_pos[0] = agent.position[0] + self.max_padding
        # agent_rel_pos[1] = agent.position[1] + self.max_padding

        # Collect visible cells as set to be plotted
        visited, rel_coords = self.field_of_view(agent.position, agent.direction, )
        local_rail_obs = None

        # Add the visible cells to the observed cells
        self.env.dev_obs_dict[handle] = set(visited)

        # Locate observed agents and their coresponding targets
        local_rail_obs = np.zeros((self.view_height, 2 * self.view_width + 1, 16))
        obs_map_state = np.zeros((self.view_height, 2 * self.view_width + 1, 2))
        obs_other_agents_state = np.zeros((self.view_height, 2 * self.view_width + 1, 4))
        _idx = 0
        for pos in visited:
            curr_rel_coord = rel_coords[_idx]
            local_rail_obs[curr_rel_coord[0], curr_rel_coord[1], :] = self.rail_obs[pos[0], pos[1], :]
            if pos == agent.target:
                obs_map_state[curr_rel_coord[0], curr_rel_coord[1], 0] = 1
            else:
                for tmp_agent in agents:
                    if pos == tmp_agent.target:
                        obs_map_state[curr_rel_coord[0], curr_rel_coord[1], 1] = 1
            if pos != agent.position:
                for tmp_agent in agents:
                    if pos == tmp_agent.position:
                        obs_other_agents_state[curr_rel_coord[0], curr_rel_coord[1], :] = np.identity(4)[
                            tmp_agent.direction]

            _idx += 1

        direction = np.identity(4)[agent.direction]
        return local_rail_obs, obs_map_state, obs_other_agents_state, direction

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[
        int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        return super().get_many(handles)

    def field_of_view(self, position, direction, state=None):
        # Compute the local field of view for an agent in the environment
        data_collection = False
        if state is not None:
            temp_visible_data = np.zeros(shape=(self.view_height, 2 * self.view_width + 1, 16))
            data_collection = True
        if direction == 0:
            origin = (position[0] + self.center, position[1] - self.view_width)
        elif direction == 1:
            origin = (position[0] - self.view_width, position[1] - self.center)
        elif direction == 2:
            origin = (position[0] - self.center, position[1] + self.view_width)
        else:
            origin = (position[0] + self.view_width, position[1] + self.center)
        visible = list()
        rel_coords = list()
        for h in range(self.view_height):
            for w in range(2 * self.view_width + 1):
                if direction == 0:
                    if 0 <= origin[0] - h < self.env.height and 0 <= origin[1] + w < self.env.width:
                        visible.append((origin[0] - h, origin[1] + w))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] - h, origin[1] + w, :]
                elif direction == 1:
                    if 0 <= origin[0] + w < self.env.height and 0 <= origin[1] + h < self.env.width:
                        visible.append((origin[0] + w, origin[1] + h))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] + w, origin[1] + h, :]
                elif direction == 2:
                    if 0 <= origin[0] + h < self.env.height and 0 <= origin[1] - w < self.env.width:
                        visible.append((origin[0] + h, origin[1] - w))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] + h, origin[1] - w, :]
                else:
                    if 0 <= origin[0] - w < self.env.height and 0 <= origin[1] - h < self.env.width:
                        visible.append((origin[0] - w, origin[1] - h))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] - w, origin[1] - h, :]
        if data_collection:
            return temp_visible_data
        else:
            return visible, rel_coords
