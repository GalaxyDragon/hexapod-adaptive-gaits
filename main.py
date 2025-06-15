import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from multiprocessing import Pool, cpu_count
import pickle
import json
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from numpy.fft import fft
from collections import defaultdict

# --- Константы и Параметры (Глобальные) ---
NUM_INTER_NEURONS = 6
NUM_MOTOR_NEURONS = 6
TOTAL_NEURONS = NUM_INTER_NEURONS + NUM_MOTOR_NEURONS
NUM_LEGS = 6
MOTOR_NEURON_INDICES = np.arange(NUM_INTER_NEURONS, TOTAL_NEURONS)
FRONT_LEFT_LEG_IDX = 0
FRONT_RIGHT_LEG_IDX = 1
MID_LEFT_LEG_IDX = 2
MID_RIGHT_LEG_IDX = 3
REAR_LEFT_LEG_IDX = 4
REAR_RIGHT_LEG_IDX = 5
LEFT_LEG_INDICES = [FRONT_LEFT_LEG_IDX, MID_LEFT_LEG_IDX, REAR_LEFT_LEG_IDX]
RIGHT_LEG_INDICES = [FRONT_RIGHT_LEG_IDX, MID_RIGHT_LEG_IDX, REAR_RIGHT_LEG_IDX]
LEFT_MOTOR_NEURONS = np.array([6, 8, 10])
RIGHT_MOTOR_NEURONS = np.array([7, 9, 11])
INTER_L_INDICES = np.array([0, 1, 2])
INTER_R_INDICES = np.array([3, 4, 5])

class Controller:
    def __init__(self, config: Dict[str, Any], links: List[Tuple[int, int]]):
        self.num_neurons = TOTAL_NEURONS
        self.num_inter_neurons = NUM_INTER_NEURONS
        self.num_motor_neurons = NUM_MOTOR_NEURONS
        self.motor_neuron_indices = MOTOR_NEURON_INDICES
        self.dt = config['simulation']['dt']
        self.tau_borders = tuple(config['controller']['tau_borders'])
        self.bias_borders = tuple(config['controller']['bias_borders'])
        self.weights_borders = tuple(config['controller']['weights_borders'])
        self.t_decay_rate = config['controller']['t_decay_rate']
        self.t_accumulation_rate = config['controller']['t_accumulation_rate']
        self.t_level_clip = tuple(config['controller']['t_level_clip'])
        self.links = links
        self.a_level = 0.0
        self.tau = np.ones(shape=self.num_neurons, dtype=np.float64) / 5
        self.bias = np.random.uniform(self.bias_borders[0], self.bias_borders[1], size=self.num_neurons)
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.weights_A = np.zeros((self.num_neurons, self.num_neurons))
        self.weights_T = np.zeros((self.num_neurons, self.num_neurons))
        for source_idx, target_idx in links:
            self.weights[target_idx, source_idx] = np.random.uniform(self.weights_borders[0]/5, self.weights_borders[1]/5)
            self.weights_A[target_idx, source_idx] = np.random.uniform(self.weights_borders[0]/5, self.weights_borders[1]/5)
            self.weights_T[target_idx, source_idx] = np.random.uniform(self.weights_borders[0]/5, self.weights_borders[1]/5)
        self.truncate()
        self.enforce_symmetry()
        self.membranes = np.zeros(self.num_neurons)
        self.outputs = np.zeros(self.num_neurons)
        self.t_level = 0.0

    def set_a_level(self, a_level: float):
        self.a_level = a_level

    def _sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def forward(self) -> np.ndarray:
        input_current = (self.outputs @ self.weights.T +
                         (self.outputs @ self.weights_A.T) * self.a_level +
                         (self.outputs @ self.weights_T.T) * self.t_level)
        dmdt = (-self.membranes + input_current) / self.tau
        self.membranes = self.membranes + dmdt * self.dt
        self.outputs = self._sigmoid(self.membranes + self.bias)
        motor_neuron_outputs = self.outputs[self.motor_neuron_indices]
        avg_motor_activity = np.mean(motor_neuron_outputs)
        self.t_level = (self.t_level * self.t_decay_rate +
                        (avg_motor_activity**2) * self.t_accumulation_rate)
        self.t_level = np.clip(self.t_level, self.t_level_clip[0], self.t_level_clip[1])
        return self.outputs

    def reset_state(self):
        self.membranes = - self.bias -0.5 
        self.outputs = np.zeros(self.num_neurons)
        self.t_level = 0.0

    def get_params_dict(self) -> Dict[str, Any]:
        return {
            'tau': self.tau.copy(),
            'bias': self.bias.copy(),
            'weights': self.weights.copy(),
            'weights_A': self.weights_A.copy(),
            'weights_T': self.weights_T.copy(),
        }

    def load_params_dict(self, params_dict: Dict[str, Any]):
        required_keys = {'tau', 'bias', 'weights', 'weights_A', 'weights_T'}
        if not required_keys.issubset(params_dict.keys()):
            raise ValueError("Отсутствуют необходимые ключи в словаре параметров.")
        if params_dict['bias'].shape != (self.num_neurons,):
            raise ValueError(f"Неверная размерность bias: {params_dict['bias'].shape}")
        self.bias = params_dict['bias']
        self.weights = params_dict['weights']
        self.weights_A = params_dict['weights_A']
        self.weights_T = params_dict['weights_T']
        self.truncate()
        self.enforce_symmetry()

    def truncate(self):
        self.tau = np.clip(self.tau, self.tau_borders[0], self.tau_borders[1])
        self.bias = np.clip(self.bias, self.bias_borders[0], self.bias_borders[1])
        self.weights = np.clip(self.weights, self.weights_borders[0], self.weights_borders[1])
        self.weights_A = np.clip(self.weights_A, self.weights_borders[0], self.weights_borders[1])
        self.weights_T = np.clip(self.weights_T, self.weights_borders[0], self.weights_borders[1])
        mask = np.zeros_like(self.weights, dtype=bool)
        if self.links:
            rows, cols = np.asarray(self.links).T
            mask[cols, rows] = True
        self.weights[~mask] = 0
        self.weights_A[~mask] = 0
        self.weights_T[~mask] = 0

    def enforce_symmetry(self):
        avg_bias = (self.bias[LEFT_MOTOR_NEURONS] + self.bias[RIGHT_MOTOR_NEURONS]) / 2.0
        self.bias[LEFT_MOTOR_NEURONS] = avg_bias
        self.bias[RIGHT_MOTOR_NEURONS] = avg_bias
        avg_w = (self.weights[LEFT_MOTOR_NEURONS, INTER_L_INDICES] + self.weights[RIGHT_MOTOR_NEURONS, INTER_R_INDICES]) / 2.0
        self.weights[LEFT_MOTOR_NEURONS, INTER_L_INDICES] = avg_w
        self.weights[RIGHT_MOTOR_NEURONS, INTER_R_INDICES] = avg_w
        avg_wa = (self.weights_A[LEFT_MOTOR_NEURONS, INTER_L_INDICES] + self.weights_A[RIGHT_MOTOR_NEURONS, INTER_R_INDICES]) / 2.0
        self.weights_A[LEFT_MOTOR_NEURONS, INTER_L_INDICES] = avg_wa
        self.weights_A[RIGHT_MOTOR_NEURONS, INTER_R_INDICES] = avg_wa
        avg_wt = (self.weights_T[LEFT_MOTOR_NEURONS, INTER_L_INDICES] + self.weights_T[RIGHT_MOTOR_NEURONS, INTER_R_INDICES]) / 2.0
        self.weights_T[LEFT_MOTOR_NEURONS, INTER_L_INDICES] = avg_wt
        self.weights_T[RIGHT_MOTOR_NEURONS, INTER_R_INDICES] = avg_wt

class HexapodState:
    def __init__(self):
        self.time: float = 0.0
        self.activations: np.ndarray = np.zeros(TOTAL_NEURONS)
        self.membranes: np.ndarray = np.zeros(TOTAL_NEURONS)
        self.T_level: float = 0.0
        self.A_level: float = 0.0
        self.leg_ups: np.ndarray = np.zeros(NUM_LEGS, dtype=bool)

class Hexapod:
    def __init__(self, controller: Controller, up_threshold: float = 0.5):
        self.controller = controller
        self.up_threshold = up_threshold
        self.num_legs = NUM_LEGS

    def step(self) -> np.ndarray:
        activations = self.controller.forward()
        motor_neuron_outputs = activations[self.controller.motor_neuron_indices]
        leg_ups = motor_neuron_outputs > self.up_threshold
        return leg_ups

    def get_state(self, time: float, leg_ups: np.ndarray) -> HexapodState:
        state = HexapodState()
        state.time = time
        state.activations = self.controller.outputs.copy()
        state.membranes = self.controller.membranes.copy()
        state.T_level = self.controller.t_level
        state.A_level = self.controller.a_level
        state.leg_ups = leg_ups.copy()
        return state

    def reset(self):
        self.controller.reset_state()

class Environment:
    def __init__(self, unit: Hexapod, config: Dict[str, Any]):
        self.unit = unit
        self.config = config
        self.max_steps = config['simulation']['max_steps_per_round']
        self.dt = config['simulation']['dt']
        self.max_unstable_steps = config['simulation'].get('max_unstable_steps_allowed', 0)
        self.min_step_down_duration = config['fitness']['min_step_down_duration']
        self.min_step_up_duration = config['fitness']['min_step_up_duration']
        self.states: List[HexapodState] = []
        self.time = 0.0
        self.current_step = 0
        self.eos = False
        self.unstable_steps_count = 0
        self.first_instability_reason = ""
        self.leg_step_counts = np.zeros(self.unit.num_legs, dtype=int)
        self.previous_leg_ups = np.zeros(self.unit.num_legs, dtype=bool)
        self.leg_phase_start_time = np.zeros(self.unit.num_legs, dtype=float)
        self.leg_up_durations: List[List[float]] = [[] for _ in range(self.unit.num_legs)]
        self.leg_down_durations: List[List[float]] = [[] for _ in range(self.unit.num_legs)]
        self.cpg_t_level_history: List[float] = []
        self.load_bearing_cost_history: List[float] = []

    def reset(self):
        self.unit.reset()
        self.time = 0.0
        self.current_step = 0
        self.states = []
        self.eos = False
        self.unstable_steps_count = 0
        self.first_instability_reason = ""
        self.leg_step_counts.fill(0)
        self.previous_leg_ups.fill(False)
        self.leg_phase_start_time.fill(0.0)
        self.leg_up_durations = [[] for _ in range(self.unit.num_legs)]
        self.leg_down_durations = [[] for _ in range(self.unit.num_legs)]
        self.cpg_t_level_history = []
        self.load_bearing_cost_history = []

    def step(self):
        if self.eos: return
        self.time += self.dt
        self.current_step += 1
        if self.current_step > self.max_steps:
            self.eos = True
            return
        leg_ups = self.unit.step()
        if self.current_step > 1:
            just_landed = ~leg_ups & self.previous_leg_ups
            just_lifted = leg_ups & ~self.previous_leg_ups
            for i in range(self.unit.num_legs):
                if just_landed[i]:
                    duration = self.time - self.leg_phase_start_time[i]
                    self.leg_up_durations[i].append(duration)
                    self.leg_phase_start_time[i] = self.time
                    self.leg_step_counts[i] += 1
                elif just_lifted[i]:
                    duration = self.time - self.leg_phase_start_time[i]
                    self.leg_down_durations[i].append(duration)
                    self.leg_phase_start_time[i] = self.time
        self.cpg_t_level_history.append(self.unit.controller.t_level)
        current_load_bearing_cost = leg_ups.sum()**2
        self.load_bearing_cost_history.append(current_load_bearing_cost)
        left_side_actives = leg_ups[LEFT_LEG_INDICES].sum()
        right_side_actives = leg_ups[RIGHT_LEG_INDICES].sum()
        total_actives = leg_ups.sum()
        front_legs_up_simultaneously = leg_ups[FRONT_LEFT_LEG_IDX] and leg_ups[FRONT_RIGHT_LEG_IDX]
        rear_legs_up_simultaneously = leg_ups[REAR_LEFT_LEG_IDX] and leg_ups[REAR_RIGHT_LEG_IDX]
        is_unstable = False
        instability_reason_current_step = ""
        if left_side_actives > 2: is_unstable = True; instability_reason_current_step = f"L>2 ({left_side_actives})"
        elif right_side_actives > 2: is_unstable = True; instability_reason_current_step = f"R>2 ({right_side_actives})"
        elif total_actives > 3: is_unstable = True; instability_reason_current_step = f"Total>3 ({total_actives})"
        elif front_legs_up_simultaneously: is_unstable = True; instability_reason_current_step = "Front Both Up"
        elif rear_legs_up_simultaneously: is_unstable = True; instability_reason_current_step = "Rear Both Up"
        if is_unstable:
            if self.current_step < 30:
                self.eos = True
                self.first_instability_reason += " (Terminated by start_from_fail)"
            self.unstable_steps_count += 1
            if not self.first_instability_reason: self.first_instability_reason = instability_reason_current_step
            if self.max_unstable_steps > 0 and self.unstable_steps_count >= self.max_unstable_steps:
                self.eos = True
                self.first_instability_reason += " (Terminated by limit)"
        current_state = self.unit.get_state(self.time, leg_ups)
        self.states.append(current_state)
        self.previous_leg_ups = leg_ups.copy()

    def get_step_duration_variance(self) -> float:
        all_valid_up_durations = [d for idx in range(self.unit.num_legs) for d in self.leg_up_durations[idx]]
        if len(all_valid_up_durations) >= 2: return np.var(all_valid_up_durations)
        else: return 0.0

    def get_average_load_bearing_cost(self) -> float:
        if self.load_bearing_cost_history: return np.sum(self.load_bearing_cost_history) * self.dt
        else: return 0.0

def calculate_rhythmicity_bonus(states: List[HexapodState], motor_indices: np.ndarray) -> float:
    num_states = len(states)
    if num_states < 20: return 0.0
    activations_history = np.array([s.activations for s in states])
    motor_activations = activations_history[:, motor_indices]
    num_steps, num_motors = motor_activations.shape
    max_magnitudes = []
    activations_variance = np.var(motor_activations, axis=1)
    for i in range(num_motors):
        signal = motor_activations[:, i] - np.mean(motor_activations[:, i])
        fft_result = fft(signal)
        spectrum = np.abs(fft_result[1:num_steps//4])
        if len(spectrum) > 0: max_magnitudes.append(np.max(spectrum))
        else: max_magnitudes.append(0.0)
    average_rhythmicity_score = np.sum(max_magnitudes)/6 if max_magnitudes else 0.0
    return average_rhythmicity_score

def calculate_fitness_components(env_results: Dict[str, Any], config: Dict[str, Any], current_a_level: float) -> Dict[str, float]:
    fitness_cfg = config['fitness']
    dt = config['simulation']['dt']
    min_steps = env_results['min_leg_steps']
    mean_steps = env_results['mean_leg_steps']
    avg_load_cost = env_results['average_load_bearing_cost']
    rhythm_score = env_results['rhythmicity_score']
    step_var = env_results['step_duration_variance']
    total_steps_run = env_results['total_steps_run']
    unstable_ratio = env_results['unstable_ratio']
    components = {}
    components['bonus_steps'] = current_a_level * (min_steps + fitness_cfg['mean_step_bonus_factor'] * mean_steps)
    components['bonus_steps'] = components['bonus_steps'] * env_results['step_len_ratio']
    components['penalty_load_bearing'] = -fitness_cfg['penalty_load_bearing_factor'] * avg_load_cost
    components['bonus_rhythm'] = fitness_cfg['rhythmicity_bonus_factor'] * rhythm_score
    components['penalty_step_variance'] = -fitness_cfg['step_duration_variance_penalty_factor'] * step_var
    components['bonus_time'] = total_steps_run * dt * fitness_cfg['time_bonus_factor']
    components['penalty_no_steps'] = -fitness_cfg['no_step_penalty'] if min_steps == 0 and total_steps_run > 0 else 0.0
    components['penalty_inactive_legs'] = -fitness_cfg['no_step_penalty'] * env_results['inactive_legs']
    components['penalty_inactive_legs'] = -fitness_cfg['no_step_penalty']* 10 *env_results['bid_step_diff']
    step_diff_penalty_val = 0
    leg_counts = env_results.get('leg_step_counts_raw', np.zeros(NUM_LEGS)) 
    if len(leg_counts) > 0 and total_steps_run > 10 and np.mean(leg_counts) > 0.5 : 
        diff = np.max(leg_counts) - np.min(leg_counts)
        if diff > 3 : 
             step_diff_penalty_val = -fitness_cfg['no_step_penalty'] * 1.5 * (diff / (np.mean(leg_counts) + 1e-6)) 
    components['penalty_step_diff_legs'] = step_diff_penalty_val
    base_fitness = sum(components.values())
    instability_penalty_power = fitness_cfg['instability_penalty_power']
    if base_fitness < 0: fitness_multiplier = (1.0 + unstable_ratio)**instability_penalty_power
    else: fitness_multiplier = (1.0 - unstable_ratio)**instability_penalty_power
    final_fitness = base_fitness * fitness_multiplier
    components['base_fitness'] = base_fitness
    components['instability_multiplier'] = fitness_multiplier
    components['final_fitness'] = final_fitness
    return components

def evaluate_agent_weighted_sum(args: Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[int, int]], List[float]]) -> Dict[str, Any]:
    controller_params_dict, config, links, a_weights = args
    a_levels_to_evaluate = config['simulation']['a_levels_to_evaluate']
    if len(a_weights) != len(a_levels_to_evaluate):
        raise ValueError("Длина a_weights должна соответствовать количеству a_levels_to_evaluate.")
    
    total_fitness = 0.0
    evaluation_details = []
    try:
        sim_controller = Controller(config, links)
        sim_controller.load_params_dict(controller_params_dict)
    except Exception as e:
        return {'params': controller_params_dict, 'fitness': -1e9, 'eval_data': [], 'error': True}

    for idx, current_a_level in enumerate(a_levels_to_evaluate):
        try:
            sim_controller.set_a_level(current_a_level)
            sim_controller.reset_state()
            robot = Hexapod(sim_controller, up_threshold=config['fitness']['leg_up_threshold'])
            env = Environment(robot, config)
            env.reset()
            while not env.eos:
                env.step()

            env_results = {
                'a_level': current_a_level,
                'min_leg_steps': env.leg_step_counts.min() if env.current_step > 0 else 0,
                'inactive_legs': (env.leg_step_counts==0).sum() if env.current_step > 0 else 6,
                'leg_step_counts_raw': env.leg_step_counts.copy(), 
                'bid_step_diff': (1 if (env.leg_step_counts.min() - env.leg_step_counts.max()) > 5 else 0) if env.current_step > 0 else 1,
                'mean_leg_steps': env.leg_step_counts.mean() if env.current_step > 0 else 0.0,
                'step_len_ratio': np.clip(np.hstack(env.leg_up_durations, dtype=np.float64)/config['fitness']['min_step_up_duration'], 0.0, 1.0).mean()**2 if env.leg_step_counts.mean() > 0 else 1.0,
                'average_load_bearing_cost': env.get_average_load_bearing_cost(),
                'step_duration_variance': env.get_step_duration_variance(),
                'total_steps_run': env.current_step,
                'unstable_steps': env.unstable_steps_count,
                'unstable_ratio': env.unstable_steps_count / env.max_steps if env.current_step > 0 else 0.0,
                'first_instability_reason': env.first_instability_reason,
                'final_cpg_t_level': env.cpg_t_level_history[-1] if env.cpg_t_level_history else 0.0,
                'total_cpg_t_level': np.sum(env.cpg_t_level_history) * env.dt if env.cpg_t_level_history else 0.0,
                'rhythmicity_score': calculate_rhythmicity_bonus(env.states, sim_controller.motor_neuron_indices),
            }
            
            fitness_components = calculate_fitness_components(env_results, config, current_a_level)
            fitness = fitness_components['final_fitness']
            weighted_fitness = a_weights[idx] * fitness
            total_fitness += weighted_fitness
            evaluation_details.append({**env_results, **fitness_components, 'weighted_fitness': weighted_fitness})
        except Exception as sim_err:
            total_fitness += a_weights[idx] * (-1e9)
            evaluation_details.append({'a_level': current_a_level, 'error': True, 'final_fitness': -1e9, 'weighted_fitness': -1e9})

    return {
        'params': controller_params_dict,
        'fitness': total_fitness,
        'eval_data': evaluation_details,
        'error': False
    }

def mutate(parent_params_dict: Dict[str, Any], config: Dict[str, Any], links: List[Tuple[int, int]]) -> Optional[Controller]:
    mutation_strength = config['genetic_algorithm']['mutation_strength']
    try:
        parent_controller = Controller(config, links)
        parent_controller.load_params_dict(parent_params_dict)
    except Exception as e: return None
    mutated_params = parent_controller.get_params_dict()
    noise_bias = np.random.normal(0, mutation_strength, size=mutated_params['bias'].shape)
    to_mutate = np.random.random(4)
    if to_mutate[3] > 0.3:
        mutated_params['bias'] += noise_bias
    noise_w = np.random.normal(0, mutation_strength, size=mutated_params['weights'].shape)
    noise_wa = np.random.normal(0, mutation_strength, size=mutated_params['weights_A'].shape)
    noise_wt = np.random.normal(0, mutation_strength, size=mutated_params['weights_T'].shape)
    mask = np.zeros_like(mutated_params['weights'], dtype=bool)
    if links:
        rows, cols = np.asarray(links).T
        mask[cols, rows] = True
    if to_mutate[0] > 0.4:
        mutated_params['weights'][mask] += noise_w[mask]
    if to_mutate[1] > 0.4:
        mutated_params['weights_A'][mask] += noise_wa[mask]
    if to_mutate[2] > 0.4:
        mutated_params['weights_T'][mask] += noise_wt[mask]
    try:
        mutated_controller = Controller(config, links)
        mutated_controller.load_params_dict(mutated_params)
        return mutated_controller
    except Exception as e: return None

def crossover(parent1_params: Dict[str, Any], parent2_params: Dict[str, Any], config: Dict[str, Any], links: List[Tuple[int, int]]) -> Optional[Controller]:
    try:
        child_controller = Controller(config, links)
        child_params = {
            'tau': (parent1_params['tau'] + parent2_params['tau']) / 2,
            'bias': (parent1_params['bias'] + parent2_params['bias']) / 2,
            'weights': (parent1_params['weights'] + parent2_params['weights']) / 2,
            'weights_A': (parent1_params['weights_A'] + parent2_params['weights_A']) / 2,
            'weights_T': (parent1_params['weights_T'] + parent2_params['weights_T']) / 2
        }
        child_controller.load_params_dict(child_params)
        child_controller.truncate()
        child_controller.enforce_symmetry()
        return child_controller
    except Exception as e:
        print(f"Ошибка кроссовера: {e}")
        return None

def get_simulation_states(args: Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[int, int]], float]) -> Tuple[List[HexapodState], int, int]:
    controller_params_dict, config, links, a_level_to_run = args
    try:
        sim_controller = Controller(config, links)
        sim_controller.load_params_dict(controller_params_dict)
        sim_controller.set_a_level(a_level_to_run)
        sim_controller.reset_state()
        robot = Hexapod(sim_controller, up_threshold=config['fitness']['leg_up_threshold'])
        env = Environment(robot, config)
        env.reset()
        while not env.eos: env.step()
        return env.states, env.unstable_steps_count, env.current_step
    except Exception as e:
        return [], 0, 0

def define_links() -> List[Tuple[int, int]]:
    links = []
    for row_start in [0, 3]:
        links.extend([(row_start + 0, row_start + 1), (row_start + 1, row_start + 0)])
        links.extend([(row_start + 1, row_start + 2), (row_start + 2, row_start + 1)])
        links.extend([(row_start + 0, row_start + 2), (row_start + 2, row_start + 0)])
    for col in range(3): links.extend([(col, col + 3), (col + 3, col)])
    links.extend([(INTER_L_INDICES[i], LEFT_MOTOR_NEURONS[i]) for i in range(3)])
    links.extend([(INTER_R_INDICES[i], RIGHT_MOTOR_NEURONS[i]) for i in range(3)])
    unique_links = sorted(list(set(links)))
    return unique_links
def train(config: Dict[str, Any]):
    """Обучает популяцию контроллеров с использованием однокритериального GA с взвешенной суммой фитнесов."""
    ga_cfg = config['genetic_algorithm']
    sim_cfg = config['simulation']
    log_cfg = config['logging']
    population_size = ga_cfg['population_size']
    generations = ga_cfg['generations']
    mutation_strength = ga_cfg['mutation_strength']
    crossover_rate = ga_cfg.get('crossover_rate', 0.8)
    initial_population_dir = ga_cfg.get('initial_population_dir', None)
    num_backups_to_save = ga_cfg.get('num_backups_to_save', 50)
    a_levels_to_evaluate = sim_cfg['a_levels_to_evaluate']
    a_weights = ga_cfg.get('a_weights', [1.0 / len(a_levels_to_evaluate)] * len(a_levels_to_evaluate))
    num_processes = sim_cfg['num_processes']
    save_interval = log_cfg['save_interval']
    plot_interval = log_cfg['plot_interval']
    base_dir_template = log_cfg['base_dir_template']

    a_level_str = "_".join([f"{a:.1f}".replace('.', 'p') for a in a_levels_to_evaluate])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = base_dir_template.format(a_levels=a_level_str, timestamp=timestamp)
    backup_dir = os.path.join(base_dir, 'backups')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    config_save_path = os.path.join(base_dir, 'config.json')
    try:
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Конфигурация сохранена в: {config_save_path}")
    except Exception as e:
        print(f"Ошибка сохранения конфигурации: {e}")
    links = define_links()

    log_file_path = os.path.join(base_dir, 'best_model_log.txt')
    with open(log_file_path, 'w') as f:
        f.write("Лог лучших моделей по поколениям\n")
        f.write(f"{'Поколение':<10} {'Общий фитнес':<15} {'Фитнес по уровням A':<50}\n")
        f.write("-" * 75 + "\n")

    population = []
    loaded_count = 0
    if initial_population_dir and os.path.isdir(initial_population_dir):
        print(f"Попытка загрузить начальную популяцию из: {initial_population_dir}")
        model_files = sorted([f for f in os.listdir(initial_population_dir) if f.endswith('.pkl')])
        for model_file in model_files:
            if len(population) >= population_size: break
            model_path = os.path.join(initial_population_dir, model_file)
            try:
                with open(model_path, 'rb') as f: params_dict = pickle.load(f)
                if 'bias' in params_dict and params_dict['bias'].shape == (TOTAL_NEURONS,):
                    population.append({'params': params_dict, 'id': len(population)})
                    loaded_count += 1
                else:
                    print(f"Пропуск несовместимой модели: {model_file}")
            except Exception as load_err:
                print(f"Не удалось загрузить {model_file}: {load_err}")
        print(f"Успешно загружено {loaded_count} индивидов.")

    needed = population_size - len(population)
    if needed > 0:
        print(f"Инициализация {needed} случайных индивидов...")
        for i in range(needed):
            controller = Controller(config=config, links=links)
            population.append({'params': controller.get_params_dict(), 'id': len(population)})

    print("--- Оценка начальной популяции P0 ---")
    eval_args = [(ind['params'], config, links, a_weights) for ind in population]
    initial_evaluation_results = []
    try:
        with Pool(processes=num_processes) as pool:
            initial_evaluation_results = pool.map(evaluate_agent_weighted_sum, eval_args)
    except Exception as e:
        print(f"Критическая ошибка при оценке P0: {e}")
        return None
    except KeyboardInterrupt:
        print("\nОбучение прервано при оценке P0.")
        return None

    for i, result in enumerate(initial_evaluation_results):
        if not result.get('error'):
            population[i].update(result)
        else:
            population[i]['fitness'] = -1e9
            population[i]['eval_data'] = []

    best_agent = max(population, key=lambda x: x['fitness'])
    fitness_per_a = [eval_data['final_fitness'] for eval_data in best_agent['eval_data']]
    with open(log_file_path, 'a') as f:
        f.write(f"{0:<10} {best_agent['fitness']:<15.4f} {str(fitness_per_a):<50}\n")

    for gen in range(generations):
        print(f"\n--- Поколение {gen+1}/{generations} ---")
        parents = []
        for _ in range(population_size):
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)
            if population[idx1]['fitness'] > population[idx2]['fitness']:
                parents.append(population[idx1])
            else:
                parents.append(population[idx2])

        offspring_population = []
        for j in range(0, population_size*4, 2):
            i = j%population_size
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < population_size else parents[i]
            if np.random.rand() < crossover_rate:
                child1 = crossover(parent1['params'], parent2['params'], config, links)
                child2 = crossover(parent2['params'], parent1['params'], config, links)
            else:
                child1 = Controller(config, links)
                child1.load_params_dict(parent1['params'])
                child2 = Controller(config, links)
                child2.load_params_dict(parent2['params'])
                child1 = mutate(child1.get_params_dict(), config, links)
                child2 = mutate(child2.get_params_dict(), config, links)
            if child1:
                offspring_population.append({'params': child1.get_params_dict(), 'id': f"g{gen+1}_o{i}"})
            if child2:
                offspring_population.append({'params': child2.get_params_dict(), 'id': f"g{gen+1}_o{i+1}"})

        offspring_eval_args = [(ind['params'], config, links, a_weights) for ind in offspring_population]
        offspring_evaluation_results = []
        try:
            with Pool(processes=num_processes) as pool:
                offspring_evaluation_results = pool.map(evaluate_agent_weighted_sum, offspring_eval_args)
        except Exception as e:
            print(f"Ошибка параллельной оценки потомков: {e}")
            continue

        for i, result in enumerate(offspring_evaluation_results):
            if not result.get('error'):
                offspring_population[i].update(result)
            else:
                offspring_population[i]['fitness'] = -1e9
                offspring_population[i]['eval_data'] = []

        combined_population = population + offspring_population
        combined_population.sort(key=lambda x: x['fitness'], reverse=True)
        population = combined_population[:population_size]

        best_agent = population[0]
        fitness_per_a = [eval_data['final_fitness'] for eval_data in best_agent['eval_data']]
        with open(log_file_path, 'a') as f:
            f.write(f"{gen+1:<10} {best_agent['fitness']:<15.4f} {str(fitness_per_a):<50}\n")
        best_model_path = os.path.join(backup_dir, f'best_gen_{gen+1}_fit_{best_agent["fitness"]:.1f}.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_agent['params'], f)

        if gen % plot_interval == 0 or gen == generations - 1:
            print("--- Детальное Логирование и Построение Графиков ---")
            gen_plots_dir = os.path.join(plots_dir, f'gen_{gen+1}')
            os.makedirs(gen_plots_dir, exist_ok=True)
            best_agent = population[0]
            agent_label = f"best_agent_gen{gen+1}"
            agent_plots_dir = os.path.join(gen_plots_dir, agent_label)
            os.makedirs(agent_plots_dir, exist_ok=True)
            print(f"\nЛогирование для {agent_label}:")
            print(f"  Общий Fitness: {best_agent['fitness']:.4f}")
            for eval_data in best_agent['eval_data']:
                a = eval_data.get('a_level', 'N/A')
                fit = eval_data.get('final_fitness', 'N/A')
                w_fit = eval_data.get('weighted_fitness', 'N/A')
                print(f"  Оценка при A={a:.2f}: Fitness={fit:.4f}, Weighted Fitness={w_fit:.4f}")
            
            for eval_data in best_agent['eval_data']:
                a_level = eval_data.get('a_level', 'N/A')
                a_str = f"{a_level:.1f}".replace('.', 'p')
                fitness_components_file = os.path.join(agent_plots_dir, f'fitness_components_A{a_str}.txt')
                with open(fitness_components_file, 'w') as f:
                    f.write(f"Компоненты фитнеса для A={a_level:.1f} (Поколение {gen+1})\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Общий фитнес: {eval_data.get('final_fitness', 0):.4f}\n")
                    f.write(f"Взвешенный фитнес: {eval_data.get('weighted_fitness', 0):.4f}\n\n")
                    f.write("Компоненты:\n")
                    f.write(f"  Бонус за шаги: {eval_data.get('bonus_steps', 0):+.4f}\n")
                    f.write(f"  Штраф за нагрузку: {eval_data.get('penalty_load_bearing', 0):+.4f}\n")
                    f.write(f"  Бонус за ритмичность: {eval_data.get('bonus_rhythm', 0):+.4f}\n")
                    f.write(f"  Штраф за вариацию длительности: {eval_data.get('penalty_step_variance', 0):+.4f}\n")
                    f.write(f"  Бонус за время: {eval_data.get('bonus_time', 0):+.4f}\n")
                    f.write(f"  Штраф за отсутствие шагов: {eval_data.get('penalty_no_steps', 0):+.4f}\n")
                    f.write(f"  Базовый фитнес: {eval_data.get('base_fitness', 0):+.4f}\n")
                    f.write(f"  Множитель нестабильности: {eval_data.get('instability_multiplier', 1):.4f}\n")
                    f.write("\nМетрики:\n")
                    f.write(f"  Минимальное число шагов: {eval_data.get('min_leg_steps', 0)}\n")
                    f.write(f"  Среднее число шагов: {eval_data.get('mean_leg_steps', 0):.2f}\n")
                    f.write(f"  Средняя нагрузка: {eval_data.get('average_load_bearing_cost', 0):.3f}\n")
                    f.write(f"  Доля нестабильных шагов: {eval_data.get('unstable_ratio', 0):.3f}\n")
                    f.write(f"  Причина нестабильности: {eval_data.get('first_instability_reason', 'Нет')}\n")
                print(f"  Компоненты фитнеса для A={a_level:.1f} сохранены в {fitness_components_file}")

            pool_args_states = [(best_agent['params'], config, links, a_level) for a_level in a_levels_to_evaluate]
            states_results = []
            try:
                with Pool(processes=num_processes) as pool_states:
                    states_results = pool_states.map(get_simulation_states, pool_args_states)
            except Exception as e:
                print(f"Ошибка при параллельном получении состояний для графиков: {e}")
                states_results = [get_simulation_states(arg) for arg in pool_args_states]

            for i, a_level_target in enumerate(a_levels_to_evaluate):
                agent_states, unstable_count, total_steps = states_results[i]
                a_str = f"{a_level_target:.1f}".replace('.', 'p')
                if agent_states:
                    time_axis = np.array([s.time for s in agent_states])
                    # График ног
                    plt.figure(figsize=(12, 6))
                    leg_ups_history = np.array([s.leg_ups for s in agent_states])
                    leg_labels = ['FL', 'FR', 'ML', 'MR', 'BL', 'BR']
                    for leg_idx in range(NUM_LEGS):
                        plt.plot(time_axis, leg_ups_history[:, leg_idx] + leg_idx * 1.1, label=f'Нога {leg_labels[leg_idx]} ({leg_idx})')
                    plt.title(f'Состояние ног ({agent_label}) - A={a_level_target:.1f}')
                    plt.xlabel('Время (с)')
                    plt.ylabel('Индекс ноги')
                    plt.yticks([])
                    plt.grid(True, axis='x')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plt.savefig(os.path.join(agent_plots_dir, f'legs_state_A{a_str}.png'))
                    plt.close()

                    # График мембран
                    plt.figure(figsize=(12, 8))
                    membranes_history = np.array([s.membranes for s in agent_states])
                    for neuron_idx in range(NUM_INTER_NEURONS):
                        plt.plot(time_axis, membranes_history[:, neuron_idx], label=f'Интер {neuron_idx}', linestyle='-', alpha=0.8)
                    for motor_idx in range(NUM_MOTOR_NEURONS):
                        plt.plot(time_axis, membranes_history[:, MOTOR_NEURON_INDICES[motor_idx]], label=f'Мото {MOTOR_NEURON_INDICES[motor_idx]}', linestyle='--', alpha=0.8)
                    plt.title(f'Мембранный потенциал ({agent_label}) - A={a_level_target:.1f}')
                    plt.xlabel('Время (с)')
                    plt.ylabel('Потенциал')
                    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
                    plt.grid(True)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plt.savefig(os.path.join(agent_plots_dir, f'membrane_potential_A{a_str}.png'))
                    plt.close()

                    # График активности моторов
                    plt.figure(figsize=(12, 6))
                    activations_history = np.array([s.activations for s in agent_states])
                    motor_activations = activations_history[:, MOTOR_NEURON_INDICES]
                    for motor_idx in range(NUM_MOTOR_NEURONS):
                        plt.plot(time_axis, motor_activations[:, motor_idx], label=f'Мото {MOTOR_NEURON_INDICES[motor_idx]}', linestyle='-', alpha=0.9)
                    plt.title(f'Активность Мотонейронов ({agent_label}) - A={a_level_target:.1f}')
                    plt.xlabel('Время (с)')
                    plt.ylabel('Активность')
                    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
                    plt.grid(True)
                    plt.ylim(-0.05, 1.05)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plt.savefig(os.path.join(agent_plots_dir, f'motor_activation_A{a_str}.png'))
                    plt.close()

                else:
                    print(f"    Предупреждение: Не удалось получить состояния для A={a_level_target:.1f}")

        # Сохранение бэкапов
        if gen % save_interval == 0 or gen == generations - 1:
            gen_backup_dir = os.path.join(backup_dir, f'gen_{gen+1}')
            os.makedirs(gen_backup_dir, exist_ok=True)
            num_to_save = min(num_backups_to_save, len(population))
            for i in range(num_to_save):
                agent_to_save = population[i]
                save_filename = os.path.join(gen_backup_dir, f'agent_{i}_fit_{agent_to_save["fitness"]:.1f}.pkl')
                with open(save_filename, 'wb') as f:
                    pickle.dump(agent_to_save['params'], f)

    print(f"\n--- Обучение Завершено ---")
    return population[0]  
def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    default_config = {
        "simulation": {
            "a_levels_to_evaluate": [1.0, 5.0],
            "dt": 0.01,
            "max_steps_per_round": 300,
            "num_processes": 0,
            "max_unstable_steps_allowed": 50
        },
        "controller": {
            "tau_borders": [0.01, 2.0],
            "bias_borders": [-3.0, 3.0],
            "weights_borders": [-5.0, 5.0],
            "t_decay_rate": 0.75,
            "t_accumulation_rate": 0.25,
            "t_level_clip": [0.0, 10.0]
        },
        "fitness": {
            "leg_up_threshold": 0.5,
            "mean_step_bonus_factor": 0.1,
            "penalty_load_bearing_factor": 0.05,
            "rhythmicity_bonus_factor": 5.0,
            "step_duration_variance_penalty_factor": 0.1,
            "time_bonus_factor": 0.01,
            "no_step_penalty": 30.0,
            "instability_penalty_power": 3.0,
            "min_step_down_duration": 0.03,
            "min_step_up_duration": 0.02
        },
        "genetic_algorithm": {
            "population_size": 100,
            "generations": 50,
            "mutation_strength": 0.1,
            "crossover_rate": 0.8,
            "initial_population_dir": None,
            "num_backups_to_save": 10,
            "a_weights": [0.5, 0.5]
        },
        "logging": {
            "save_interval": 10,
            "plot_interval": 10,
            "base_dir_template": "./ga_results_{timestamp}_A{a_levels}"
        }
    }
    config = default_config
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f: loaded_config = json.load(f)
            print(f"Конфигурация загружена из: {config_path}")
            def merge_configs(default, loaded):
                merged = default.copy()
                for key, value in loaded.items():
                    if isinstance(value, dict) and isinstance(merged.get(key), dict): merged[key] = merge_configs(merged[key], value)
                    else: merged[key] = value
                return merged
            config = merge_configs(default_config, loaded_config)
        except Exception as e: print(f"Ошибка загрузки конфигурации: {e}. Используется дефолтная."); config = default_config
    try:
        max_cpu = cpu_count()
        if config['simulation']['num_processes'] <= 0: config['simulation']['num_processes'] = max_cpu
        else: config['simulation']['num_processes'] = min(config['simulation']['num_processes'], max_cpu)
    except NotImplementedError: config['simulation']['num_processes'] = 1
    return config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обучение CPG контроллера (GA с взвешенной суммой).")
    parser.add_argument('-c', '--config', type=str, default='./sologa.json', help="Путь к файлу конфигурации JSON.")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Используется {config['simulation']['num_processes']} процессов.")
    best_solution = train(config)
    if best_solution:
        print(f"\n--- Найдено лучшее решение ---")
        print(f"Фитнес (взвешенная сумма): {best_solution['fitness']:.4f}")
        base_dir = sorted([d for d in os.listdir('.') if d.startswith('./ga_results_')])[-1]
        final_path = os.path.join(base_dir, 'best_solution.pkl')
        with open(final_path, 'wb') as f:
            pickle.dump(best_solution['params'], f)
        print(f"Лучшее решение сохранено в {final_path}")
