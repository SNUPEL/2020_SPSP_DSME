import os
import numpy as np
from time import time


def random_stacking(state, mask):
    state = state.squeeze(axis=0)
    mask = mask.astype(bool)
    actions = np.array([i for i in range(state.shape[-1] - 1)])
    action = np.random.choice(actions[mask])
    return action


def minimize_conflicts(state, mask):
    state = state.squeeze(axis=0)
    mask_action = mask.astype(bool)
    actions = np.array([i for i in range(state.shape[-1] - 1)])

    ref = np.min(np.where(state[:, 0] != -1.0)[0][0])
    state[state == -1.0] = np.max(state)

    E = np.min(state[:, 1:], axis=0)
    mask_final = list(E >= state[ref, 0])
    mask_final = mask_final & mask_action
    if sum(mask_final) == 0:
        action = np.random.choice(actions[mask_action])
    else:
        action = np.random.choice(actions[mask_final])

    return action


def delay_conflicts(state, mask):
    state = state.squeeze(axis=0)
    mask_action = mask.astype(bool)
    actions = np.array([i for i in range(state.shape[-1] - 1)])

    ref = np.min(np.where(state[:, 0] != -1.0)[0][0])
    state[state == -1.0] = np.max(state)

    E = np.min(state[:, 1:], axis=0)
    mask_final = list(E >= state[ref, 0])
    mask_final = mask_final & mask_action
    if sum(mask_final) == 0:
        mask_final = (E == max(E[mask_action])) & mask_action
        action = np.random.choice(actions[mask_final])
    else:
        action = np.random.choice(actions[mask_final])

    return action


def flexibility_optimization(state, mask):
    state = state.squeeze(axis=0)
    mask_action = mask.astype(bool)
    actions = np.array([i for i in range(state.shape[-1] - 1)])

    ref = np.min(np.where(state[:, 0] != -1.0)[0][0])
    state[state == -1.0] = np.max(state)

    E = np.min(state[:, 1:], axis=0)
    dF = state[ref, 0] - E
    mask_final = (dF <= 0)
    mask_final = mask_final & mask_action
    if sum(mask_final) == 0:
        mask_final = (dF == min(dF[mask_action])) & mask_action
        action = np.random.choice(actions[mask_final])
    else:
        mask_final = (dF == max(dF[mask_final])) & mask_action
        action = np.random.choice(actions[mask_final])

    return action