from chainer import Variable
from chainer import functions as F

import numpy as np
import matplotlib.pyplot as plt


class L_Norm_Attacks(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_minus_one_idxs(state):
        minus_one_indexes = []
        for i, val in enumerate(state):
            if val == -1:
                minus_one_indexes.append(i)
        return minus_one_indexes

    @staticmethod
    def _process_pertubed_state(curr_state, original_state):
        perturbed_state = F.clip(curr_state, 0, 1).data
        minus_one_indexes = L_Norm_Attacks._get_minus_one_idxs(original_state)
        np.put(perturbed_state, minus_one_indexes, [-1] * len(minus_one_indexes))
        perturbed_state = perturbed_state.squeeze(0)
        return perturbed_state

    @staticmethod
    def determenistic_fsgm_attack(state, model, target, epsilon=0.2, loss='Logits'):

        bs = np.expand_dims(state, 0)
        delta = np.zeros_like(bs)
        # action_distrib, action_value, v = model(bs)
        # print("Numpy:")
        # print(action_distrib, action_value, v)
        # print(action_distrib.most_probable.array[0])

        bs_chainer = Variable(data=bs, requires_grad=False)
        delta_chainer = Variable(data=delta, requires_grad=True)
        state_to_eval = bs_chainer + delta_chainer

        action_distrib, action_value, v = model(state_to_eval)

        logits_action_dist = action_distrib.logits
        # softmax_action_dist = action_distrib.all_prob

        if loss == 'Logits':
            loss = 2 * logits_action_dist[:, target] - F.sum(logits_action_dist, axis=1)
        else:
            target_label = np.array([target]).astype(np.int32)
            loss = -1 * F.softmax_cross_entropy(logits_action_dist, target_label)

        loss = F.squeeze(loss)
        loss.backward()

        perturbed_state = L_Norm_Attacks._process_pertubed_state(state + epsilon * F.sign(delta_chainer.grad_var),
                                                                 state)
        return perturbed_state

    @staticmethod
    def determenistic_pgd_attack(state, model, target, epsilon=0.2, loss_f='Logits', num_iter=5):

        alpha = epsilon / num_iter

        bs = np.expand_dims(state, 0)
        delta = np.zeros_like(bs)
        # action_distrib, action_value, v = model(bs)
        # print("Numpy:")
        # print(action_distrib, action_value, v)
        # print(action_distrib.most_probable.array[0])

        perturbed_state = bs
        coefficient = sum([1 / (t + 1) for t in range(1, num_iter + 1)])
        for t in range(1, num_iter + 1):
            alpha = epsilon * (1 / ((t + 1) * coefficient))
            # print("started iter: "+str(t+1)+"!")
            bs_chainer = Variable(data=perturbed_state, requires_grad=False)
            delta_chainer = Variable(data=delta, requires_grad=True)
            state_to_eval = bs_chainer + delta_chainer

            action_distrib, action_value, _ = model(state_to_eval)

            logits_action_dist = action_distrib.logits
            # softmax_action_dist = action_distrib.all_prob

            if loss_f == 'Logits':
                loss = 2 * logits_action_dist[:, target] - F.sum(logits_action_dist, axis=1)
            else:
                target_label = np.array([target]).astype(np.int32)
                loss = -1 * F.softmax_cross_entropy(logits_action_dist, target_label)

            loss = F.squeeze(loss)
            loss.backward()
            # print(delta_chainer.grad)

            nonclip_val = delta_chainer.data + alpha * F.sign(delta_chainer.grad_var)
            delta_chainer.cleargrad()
            delta = F.clip(nonclip_val, -epsilon, epsilon).data

        perturbed_state = L_Norm_Attacks._process_pertubed_state(bs_chainer + delta, state)

        return perturbed_state

    @staticmethod
    def misclassify_determenistic_exhaustive_targeted_linf_gradient(url, env, agent, target, epsilon=0.2,
                                                                    loss_f='Logits', attack='pgd', step_eps='uniform'):

        def _run_traj_helper(url, env, agent, states):
            # print("*start_traj*")
            # print("input:")
            # print(states)
            # print("others;")
            reward, done = 0, False
            state = env.reset(url=url)
            action = agent.act(state)
            state, _, done, _info = env.step(action)

            for i in range(len(states)):
                env.set_pertubed_state(states[i])
                action = agent.act(states[i])
                # print(action)
                state, reward, done, _info = env.step(action)
                if done:
                    break
            # print(state)
            if len(states) > 0:
                env.set_pertubed_state(states[-1])  # revert last env.step

            # print("arr_state:"+str(states[-1]))
            # print("env_state:" + str(env.state))
            # print("Done:"+str(done))
            # print("*end_traj*")
            return state, reward, done, _info, action

        def attack_traversal_traj(url, env, agent, p_states, curr_target, target, epsilon, loss_f, attack, step_eps_lst,
                                  step_eps_idx):

            model = agent.get_model()

            state, reward, done, _info, last_action = _run_traj_helper(url, env, agent, p_states)

            success_attack = False

            if len(p_states) == 0:
                pertubed_states = []
            else:
                pertubed_states = p_states.copy()

            if done:  # If I got here, it means there is no attack
                agent.stop_episode()
                # print("*start-done-traj*")
                # print(last_action)
                if last_action == target:
                    success_attack = True
                # print(success_attack)
                # print("*end-done-traj*")
                return success_attack, pertubed_states

            pertubed_state = None
            if step_eps_idx == -1:  # full l_inf at once, not use weighted epsilon
                # print("in -1")
                if attack == 'pgd':
                    pertubed_state = np.copy(
                        L_Norm_Attacks.determenistic_pgd_attack(np.copy(state), model, target, epsilon, loss_f))
                elif attack == 'fsgm':
                    pertubed_state = np.copy(
                        L_Norm_Attacks.determenistic_fsgm_attack(np.copy(state), model, target, epsilon, loss_f))

                pertubed_states.append(np.copy(pertubed_state))
                _state = np.copy(pertubed_state)
                while not done:
                    action = agent.act(_state)
                    _state, reward, done, _info = env.step(action)
                    if not done:
                        pertubed_states.append(np.copy(_state))
                agent.stop_episode()
                if action == target:
                    success_attack = True
                # print("action:"+str(action)+', target:'+str(target))
                # print(success_attack)
                # print("out -1")
                return success_attack, pertubed_states

            else:  # do traj search with decayed/uniform epsilon
                # print("index: "+str(step_eps_idx))
                # print("curr_eps: "+str(step_eps_lst[step_eps_idx]))
                if attack == 'pgd':
                    pertubed_state = np.copy(L_Norm_Attacks.determenistic_pgd_attack(np.copy(state), model, curr_target,
                                                                                     step_eps_lst[step_eps_idx],
                                                                                     loss_f))
                elif attack == 'fsgm':
                    pertubed_state = np.copy(
                        L_Norm_Attacks.determenistic_fsgm_attack(np.copy(state), model, curr_target,
                                                                 step_eps_lst[step_eps_idx], loss_f))

                curr_action = agent.act(pertubed_state)
                # print("in decay!")
                # print("curr_action:" +str(curr_action)+', curr_target:'+str(curr_target)+', target:'+str(target))
                if curr_action == curr_target:  # succeeed attack
                    # print("curr_action == curr_target")
                    new_states = pertubed_states + [np.copy(pertubed_state)]
                    success_attack_after, pertubed_states_after = attack_traversal_traj(url, env, agent, new_states,
                                                                                        target, target, epsilon, loss_f,
                                                                                        attack, step_eps_lst,
                                                                                        step_eps_idx + 1)
                    # print("out " + str(step_eps_idx))
                    return success_attack_after, pertubed_states_after
                elif curr_target != target:  # avoid continuing when not get other wanted action/detector
                    # print("curr_target != target")
                    # agent.stop_episode()
                    # print("out " + str(step_eps_idx))
                    return success_attack, pertubed_states
                else:  # attack to target failed, need to check other detectors
                    # print("try other detector")
                    num_of_detectors = state.shape[0]
                    not_used_detectors = [x for x in range(num_of_detectors) if state[x] == -1]

                    for ac in not_used_detectors:
                        # print("trying other detector:"+str(ac))
                        success_attack_after, pertubed_states_after = attack_traversal_traj(url, env, agent,
                                                                                            pertubed_states, ac, target,
                                                                                            epsilon, loss_f, attack,
                                                                                            step_eps_lst, step_eps_idx)
                        if success_attack_after == True:  # found attack
                            success_attack = True
                            pertubed_states = pertubed_states_after.copy()
                            return success_attack, pertubed_states
                    agent.stop_episode()
                    # print("out " + str(step_eps_idx))
                    return success_attack, pertubed_states

        done = False
        state = env.reset(url=url)
        states_real = []
        while not done:
            action = agent.act(state)
            state, reward, done, _info = env.step(action)
            if not done:
                states_real.append(np.copy(state))
        agent.stop_episode()

        num_of_detectors = state.shape[0]
        assert num_of_detectors > 0

        if step_eps == 'uniform':
            step_eps_lst = [[epsilon / num_of_detectors] * num_of_detectors]
        elif step_eps == 'decay':
            step_eps_lst = [[epsilon * val for val in [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]]]
        elif step_eps == 'grow':
            step_eps_lst = [[epsilon * val for val in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]]]
        else:  # UNIFIED
            step_eps_lst = [[epsilon / num_of_detectors] * num_of_detectors] + [
                [epsilon * val for val in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]]] + [
                               [epsilon * val for val in [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]]]

        success_attack, pertubed_states = False, []
        num_detector_changes = 0
        magnitude_changes = np.full(shape=(num_of_detectors,), fill_value=0, dtype=np.float32)
        if len(states_real) == 0:
            return success_attack, pertubed_states, num_detector_changes, magnitude_changes

        for i in range(len(states_real)):
            for eps_lst in step_eps_lst:
                # print("***")
                # print(states_real[:i])
                success_attack, pertubed_states = attack_traversal_traj(url, env, agent, states_real[:i], target,
                                                                        target, epsilon, loss_f, attack, eps_lst,
                                                                        -1)  # full attack ,-1
                if success_attack == False:
                    success_attack, pertubed_states = attack_traversal_traj(url, env, agent, states_real[:i], target,
                                                                            target, epsilon, loss_f, attack, eps_lst,
                                                                            0)  # weighted attack, start from 0 idx
                if success_attack == True:
                    original_state = states_real[-1]
                    pertubed_state = pertubed_states[-1]
                    for i in range(num_of_detectors):
                        if original_state[i] == -1 or pertubed_state[i] == -1:
                            if original_state[i] != pertubed_state[i]:
                                num_detector_changes += 1
                        else:
                            magnitude = np.abs(original_state[i] - pertubed_state[i])
                            magnitude_changes[i] = magnitude
                    return success_attack, pertubed_states, num_detector_changes, magnitude_changes
        return success_attack, pertubed_states, num_detector_changes, magnitude_changes

    @staticmethod
    def misclassify_determenistic_exhaustive_targeted_non_gradient(url, env, agent, target, epsilon=0.2, attack='l1'):

        def _run_traj_helper(url, env, agent, states):
            reward, done = 0, False
            state = env.reset(url=url)
            action = agent.act(state)
            state, _, done, _info = env.step(action)

            for i in range(len(states)):
                env.set_pertubed_state(states[i])
                action = agent.act(states[i])
                # print(action)
                state, reward, done, _info = env.step(action)
                if done:
                    break
            # print(state)
            if len(states) > 0:
                env.set_pertubed_state(states[-1])  # revert last env.step

            # print("arr_state:"+str(states[-1]))
            # print("env_state:" + str(env.state))
            # print("Done:"+str(done))
            # print("*end_traj*")
            return state, reward, done, _info, action

        def pertube_state_helper(state, action, step_eps_idx_lst, eps_lst, add_or_sub='add'):

            MAX_VAL = 1
            MIN_VAL = 0

            eps_idx = step_eps_idx_lst[action]
            curr_eps = eps_lst[eps_idx]
            pertubed_state = np.copy(state)
            if add_or_sub == 'add':
                new_val = min(pertubed_state[action] + curr_eps, MAX_VAL)
            else:
                new_val = max(pertubed_state[action] - curr_eps, MIN_VAL)
            pertubed_state[action] = new_val
            return pertubed_state

        def attack_traversal_traj(url, env, agent, p_states, target, eps_lst, step_eps_idx_lst, attack='l1'):

            state, reward, done, _info, last_action = _run_traj_helper(url, env, agent, p_states)

            success_attack = False

            if len(p_states) == 0:
                pertubed_states = []
            else:
                pertubed_states = p_states.copy()

            if done:  # If I got here, it means there is no attack
                agent.stop_episode()
                if last_action == target:
                    success_attack = True
                return success_attack, pertubed_states

            pertubed_state = None
            num_of_detectors = state.shape[0]
            used_detectors = [x for x in range(num_of_detectors) if state[x] != -1]
            for ac in used_detectors:
                if attack == 'linf' and step_eps_idx_lst[
                    ac] > 0:  # avoid choosing to pertube this detector again and give another eps. in L1 it is okay, since it would be eps/2 + eps/4 ...
                    continue
                for val in ['add', 'sub']:  # add epsilon or subtract
                    new_states = []
                    pertubed_state = pertube_state_helper(state, ac, step_eps_idx_lst, eps_lst, add_or_sub=val)
                    new_states.append(np.copy(pertubed_state))
                    _state = np.copy(pertubed_state)
                    while not done:
                        action = agent.act(_state)
                        _state, reward, done, _info = env.step(action)
                        if not done:
                            new_states.append(np.copy(_state))
                    agent.stop_episode()
                    if action == target:
                        success_attack = True
                        return success_attack, pertubed_states + new_states
                    else:
                        step_eps_idx_lst_forward = np.copy(step_eps_idx_lst)
                        if attack == 'l1':
                            for j in range(len(step_eps_idx_lst_forward)):
                                step_eps_idx_lst_forward[j] = step_eps_idx_lst_forward[j] + 1
                        else:  # linf
                            step_eps_idx_lst_forward[ac] = step_eps_idx_lst_forward[ac] + 1

                        success_attack_after, pertubed_states_after = attack_traversal_traj(url, env, agent,
                                                                                            p_states + [np.copy(
                                                                                                pertubed_state)],
                                                                                            target, eps_lst,
                                                                                            step_eps_idx_lst_forward,
                                                                                            attack)
                        if success_attack_after == True:
                            success_attack = True
                            pertubed_states = pertubed_states_after.copy()
                            return success_attack, pertubed_states
            return success_attack, pertubed_states

        done = False
        state = env.reset(url=url)
        states_real = []
        while not done:
            action = agent.act(state)
            state, reward, done, _info = env.step(action)
            if not done:
                states_real.append(np.copy(state))
        agent.stop_episode()

        num_of_detectors = state.shape[0]
        assert num_of_detectors > 0

        if attack == 'l1':
            eps_lst_of_lists = [[epsilon / i] * i + [0] * (num_of_detectors - i) for i in
                                range(1, num_of_detectors + 1)]
        elif attack == 'linf':
            eps_lst_of_lists = [[epsilon] * num_of_detectors]

        idx_used_attacks_lst = np.full(shape=(num_of_detectors,), fill_value=0, dtype=np.int32)

        success_attack, pertubed_states = False, []
        num_detector_changes = 0
        magnitude_changes = np.full(shape=(num_of_detectors,), fill_value=0, dtype=np.float32)
        if len(states_real) == 0:
            return success_attack, pertubed_states, num_detector_changes, magnitude_changes
        for eps_lst in eps_lst_of_lists:
            for i in range(len(states_real)):
                success_attack, pertubed_states = attack_traversal_traj(url, env, agent, states_real[:i], target,
                                                                        eps_lst, idx_used_attacks_lst,
                                                                        attack)  # full attack ,-1
                if success_attack == True:
                    original_state = states_real[-1]
                    pertubed_state = pertubed_states[-1]
                    for i in range(num_of_detectors):
                        if original_state[i] == -1 or pertubed_state[i] == -1:
                            if original_state[i] != pertubed_state[i]:
                                num_detector_changes += 1
                        else:
                            magnitude = np.abs(original_state[i] - pertubed_state[i])
                            magnitude_changes[i] = magnitude
                    return success_attack, pertubed_states, num_detector_changes, magnitude_changes
        return success_attack, pertubed_states, num_detector_changes, magnitude_changes

    @staticmethod
    def resources_determenistic_exhaustive_targeted_linf_gradient(url, env, agent, target, epsilon=0.2, loss_f='Logits',
                                                                  attack='pgd', step_eps='uniform', mode='len'):
        '''

        :param url:
        :param env:
        :param agent:
        :param target:
        :param epsilon:
        :param loss_f:
        :param attack:
        :param step_eps:
        :param mode: len-attacking more detectors, time- longer processing time, both- both of them is successfull attack,otherwise not
        :return: True if successfull attack
        '''

        def get_num_curr_detectors(state):
            num_curr_detectors = sum([1 for detector_pred in state if detector_pred != -1])
            return num_curr_detectors

        def get_processing_time(url, env, state):
            curr_processing_time = 0
            for idx, pred in enumerate(state):
                if pred != -1:
                    _, curr_cost = env.apply_detectors(idx, url)
                    curr_cost_by_func = env.calc_func_cost(curr_cost)
                    curr_processing_time += curr_cost_by_func
            return curr_processing_time

        def check_attack_requirements(url, env, state, mode, num_real_detectors, real_proccessing_time):
            num_curr_detectors = get_num_curr_detectors(state)
            curr_proccessing_time = get_processing_time(url, env, state)
            num_detectors_cond = num_real_detectors < num_curr_detectors
            proccessing_cond = real_proccessing_time < curr_proccessing_time
            if mode == 'len':
                return num_detectors_cond
            elif mode == 'time':
                return proccessing_cond
            else:  # both
                return num_detectors_cond and proccessing_cond

        def _run_traj_helper(url, env, agent, states):
            # print("*start_traj*")
            # print("input:")
            # print(states)
            # print("others;")
            reward, done = 0, False
            state = env.reset(url=url)
            action = agent.act(state)
            state, _, done, _info = env.step(action)

            for i in range(len(states)):
                env.set_pertubed_state(states[i])
                action = agent.act(states[i])
                # print(action)
                state, reward, done, _info = env.step(action)
                if done:
                    break
            # print(state)
            if len(states) > 0:
                env.set_pertubed_state(states[-1])  # revert last env.step

            # print("arr_state:"+str(states[-1]))
            # print("env_state:" + str(env.state))
            # print("Done:"+str(done))
            # print("*end_traj*")
            return state, reward, done, _info, action

        def attack_traversal_traj(url, env, agent, p_states, curr_target, target, epsilon, loss_f, attack, step_eps_lst,
                                  step_eps_idx, mode, num_real_detectors=-1, real_proccessing_time=-1):

            model = agent.get_model()

            state, reward, done, _info, last_action = _run_traj_helper(url, env, agent, p_states)

            success_attack = False

            if len(p_states) == 0:
                pertubed_states = []
            else:
                pertubed_states = p_states.copy()

            if done:
                agent.stop_episode()
                # print("*start-done-traj*")
                # print(last_action)
                if len(pertubed_states) == 0:  # avoid direct action
                    return success_attack, pertubed_states
                curr_condition = check_attack_requirements(url, env, pertubed_states[-1], mode, num_real_detectors,
                                                           real_proccessing_time)
                if last_action == target and curr_condition is True:
                    success_attack = True
                # print(success_attack)
                # print("*end-done-traj*")
                return success_attack, pertubed_states

            pertubed_state = None
            if step_eps_idx == -1:  # full l_inf at once, not use weighted epsilon
                # print("in -1")
                if attack == 'pgd':
                    pertubed_state = np.copy(
                        L_Norm_Attacks.determenistic_pgd_attack(np.copy(state), model, target, epsilon, loss_f))
                elif attack == 'fsgm':
                    pertubed_state = np.copy(
                        L_Norm_Attacks.determenistic_fsgm_attack(np.copy(state), model, target, epsilon, loss_f))

                pertubed_states.append(np.copy(pertubed_state))
                _state = np.copy(pertubed_state)
                while not done:
                    action = agent.act(_state)
                    _state, reward, done, _info = env.step(action)
                    if not done:
                        pertubed_states.append(np.copy(_state))
                agent.stop_episode()
                curr_condition = check_attack_requirements(url, env, _state, mode, num_real_detectors,
                                                           real_proccessing_time)
                if action == target and curr_condition is True:
                    success_attack = True
                # print("action:"+str(action)+', target:'+str(target))
                # print(success_attack)
                # print("out -1")
                return success_attack, pertubed_states

            else:  # do traj search with decayed/uniform epsilon
                # print("index: "+str(step_eps_idx))
                # print("curr_eps: "+str(step_eps_lst[step_eps_idx]))
                if attack == 'pgd':
                    pertubed_state = np.copy(L_Norm_Attacks.determenistic_pgd_attack(np.copy(state), model, curr_target,
                                                                                     step_eps_lst[step_eps_idx],
                                                                                     loss_f))
                elif attack == 'fsgm':
                    pertubed_state = np.copy(
                        L_Norm_Attacks.determenistic_fsgm_attack(np.copy(state), model, curr_target,
                                                                 step_eps_lst[step_eps_idx], loss_f))

                curr_action = agent.act(pertubed_state)
                # print("in decay!")
                # print("curr_action:" +str(curr_action)+', curr_target:'+str(curr_target)+', target:'+str(target))
                if curr_action == curr_target:  # succeeed attack
                    # print("curr_action == curr_target")
                    new_states = pertubed_states + [np.copy(pertubed_state)]
                    success_attack_after, pertubed_states_after = attack_traversal_traj(url, env, agent, new_states,
                                                                                        target, target, epsilon, loss_f,
                                                                                        attack, step_eps_lst,
                                                                                        step_eps_idx + 1, mode,
                                                                                        num_real_detectors,
                                                                                        real_proccessing_time)
                    # print("out " + str(step_eps_idx))
                    return success_attack_after, pertubed_states_after
                elif curr_target != target:  # avoid continuing when not get other wanted action/detector
                    # print("curr_target != target")
                    # agent.stop_episode()
                    # print("out " + str(step_eps_idx))
                    return success_attack, pertubed_states
                else:  # attack to target failed, need to check other detectors
                    # print("try other detector")
                    num_of_detectors = state.shape[0]
                    not_used_detectors = [x for x in range(num_of_detectors) if state[x] == -1]

                    for ac in not_used_detectors:
                        # print("trying other detector:"+str(ac))
                        success_attack_after, pertubed_states_after = attack_traversal_traj(url, env, agent,
                                                                                            pertubed_states, ac, target,
                                                                                            epsilon, loss_f, attack,
                                                                                            step_eps_lst, step_eps_idx,
                                                                                            mode, num_real_detectors,
                                                                                            real_proccessing_time)
                        if success_attack_after == True:  # found attack
                            success_attack = True
                            pertubed_states = pertubed_states_after.copy()
                            return success_attack, pertubed_states
                    agent.stop_episode()
                    # print("out " + str(step_eps_idx))
                    return success_attack, pertubed_states

        done = False
        state = env.reset(url=url)
        states_real = []
        while not done:
            action = agent.act(state)
            state, reward, done, _info = env.step(action)
            if not done:
                states_real.append(np.copy(state))
        agent.stop_episode()

        num_of_detectors = state.shape[0]
        assert num_of_detectors > 0

        if step_eps == 'uniform':
            step_eps_lst = [[epsilon / num_of_detectors] * num_of_detectors]
        elif step_eps == 'decay':
            step_eps_lst = [[epsilon * val for val in [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]]]
        elif step_eps == 'grow':
            step_eps_lst = [[epsilon * val for val in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]]]
        else:  # UNIFIED
            step_eps_lst = [[epsilon / num_of_detectors] * num_of_detectors] + [
                [epsilon * val for val in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]]] + [
                               [epsilon * val for val in [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]]]

        success_attack, pertubed_states = False, []
        num_detector_changes = 0
        magnitude_changes = np.full(shape=(num_of_detectors,), fill_value=0, dtype=np.float32)

        if len(states_real) == 0:
            return success_attack, pertubed_states, num_detector_changes, magnitude_changes

        num_real_detectors = get_num_curr_detectors(states_real[-1])
        real_proccessing_time = get_processing_time(url, env, states_real[-1])

        for i in range(len(states_real)):
            for eps_lst in step_eps_lst:
                # print("***")
                # print(states_real[:i])
                success_attack, pertubed_states = attack_traversal_traj(url, env, agent, states_real[:i], target,
                                                                        target, epsilon, loss_f, attack, eps_lst, -1,
                                                                        mode, num_real_detectors,
                                                                        real_proccessing_time)  # full attack ,-1
                if success_attack == False:
                    success_attack, pertubed_states = attack_traversal_traj(url, env, agent, states_real[:i], target,
                                                                            target, epsilon, loss_f, attack, eps_lst, 0,
                                                                            mode, num_real_detectors,
                                                                            real_proccessing_time)  # weighted attack, start from 0 idx
                if success_attack == True:
                    original_state = states_real[-1]
                    pertubed_state = pertubed_states[-1]
                    for i in range(num_of_detectors):
                        if original_state[i] == -1 or pertubed_state[i] == -1:
                            if original_state[i] != pertubed_state[i]:
                                num_detector_changes += 1
                        else:
                            magnitude = np.abs(original_state[i] - pertubed_state[i])
                            magnitude_changes[i] = magnitude
                    return success_attack, pertubed_states, num_detector_changes, magnitude_changes
        return success_attack, pertubed_states, num_detector_changes, magnitude_changes

    @staticmethod
    def determenistic_exhaustive_non_gradient_baselines(model, sample, detectors, target, label, epsilon, attack='l1',
                                                        threshold=0.5, lib='sklearn'):

        def cast_by_threshold(val):
            result = 1
            if val < threshold:
                result = 0
            return result

        def get_right_eps(curr_vector, original_vector):
            idx = 0
            for i in range(len(original_vector)):
                if original_vector[i] != curr_vector[i]:
                    idx += 1
            return idx

        success_attack = False
        pertubed_vector = None
        start_vector = None
        magnitude_changes = None

        if label != target:

            MAX_VAL = 1
            MIN_VAL = 0
            num_of_detectors = len(detectors)

            if attack == 'l1':
                eps_lst_of_lists = [[epsilon / i] * i + [0] * (num_of_detectors - i) for i in
                                    range(1, num_of_detectors + 1)]

            elif attack == 'linf':
                eps_lst_of_lists = [[epsilon] * num_of_detectors]

            for epsilons in eps_lst_of_lists:

                start_vector = np.full(shape=(num_of_detectors,), fill_value=-1, dtype=np.float32)
                for i in range(num_of_detectors):
                    result = sample[detectors[i]]
                    start_vector[i] = result
                vectors = [start_vector]  # added clean vector to create attacks from it

                for i in range(num_of_detectors):
                    new_vectors = []
                    for _vector in vectors:
                        curr_idx = get_right_eps(_vector, start_vector)
                        eps = epsilons[curr_idx]
                        if eps == 0:
                            continue

                        new_val_up = min(_vector[i] + eps, MAX_VAL)
                        vector_cpy_up = np.copy(_vector)
                        vector_cpy_up[i] = new_val_up
                        new_vectors.append(vector_cpy_up)

                        new_val_down = max(_vector[i] - eps, MIN_VAL)
                        vector_cpy_down = np.copy(_vector)
                        vector_cpy_down[i] = new_val_down
                        new_vectors.append(vector_cpy_down)

                    vectors += new_vectors

                vectors.pop(0)  # avoid trying clean vector

                vectors_npy = np.array(vectors)

                if lib == 'sklearn':
                    preds = model.predict(vectors_npy)
                else:
                    preds = model(vectors_npy)
                for i, pred in enumerate(preds):
                    casted_pred = cast_by_threshold(pred)
                    if casted_pred == target:
                        success_attack = True
                        pertubed_vector = np.copy(vectors_npy[i])
                        magnitude_changes = np.full(shape=(num_of_detectors,), fill_value=0, dtype=np.float32)
                        for i in range(num_of_detectors):
                            magnitude = np.abs(start_vector[i] - pertubed_vector[i])
                            magnitude_changes[i] = magnitude
                        return success_attack, pertubed_vector, start_vector, magnitude_changes

        return success_attack, pertubed_vector, start_vector, magnitude_changes

    @staticmethod
    def misclassify_attack_plotting_vectorized(url, env, agent, target, order=np.inf, opposite=False, dim=2,
                                               max_epsilon=1, to_plot=False, resolution=10):
        @np.vectorize
        def vectorized_traj(x, y):
            p_state = np.copy(pertube_state)
            p_state[curr_indexes[0]] = x
            p_state[curr_indexes[1]] = y

            p_done = False
            _ = env.reset(url=url)
            env.set_pertubed_state(p_state)

            p_action = -1
            while not p_done:
                p_action = agent.act(p_state)
                p_state, _, p_done, _ = env.step(p_action)
            agent.stop_episode()
            return int(p_action == target)

        done = False
        state = env.reset(url=url)
        states_real = []
        while not done:
            action = agent.act(state)
            state, reward, done, _info = env.step(action)
            if not done:
                states_real.append(np.copy(state))
        agent.stop_episode()

        real_label = 'b' if url['label'] == 0 else 'p'

        # if len(states_real) < 3: #avoid undesired cases
        #   return []

        # title_str = f"{url['url']} \n "

        shortened_state_real = states_real[:dim] if len(states_real) >= dim else states_real
        list_of_eps_dims = []
        for _state in shortened_state_real:
            idx_states = list(np.where(_state > -1)[0])
            idx_states = [idx_states[0]] * 2 if len(idx_states) == 1 else idx_states
            list_of_eps_per_dim = [5]  # 5 symbolize of not found attack, if we min after and it stays 5
            for i in range(len(idx_states)):
                for j in range(i + 1, len(idx_states)):
                    original_state = np.copy(_state)
                    pertube_state = np.copy(_state)
                    curr_indexes = [idx_states[i], idx_states[j]]

                    pertube_str = list(np.copy(_state))
                    pertube_str[curr_indexes[1]] = 'y axis'
                    pertube_str[curr_indexes[0]] = 'x axis' if curr_indexes[0] != curr_indexes[1] else 'x and y axes'

                    if opposite is True:
                        for idx in set(idx_states) - set(curr_indexes):
                            if pertube_state[idx] < 0.5:
                                pertube_state[idx] = min(pertube_state[idx] + max_epsilon, 1)
                            else:
                                pertube_state[idx] = max(pertube_state[idx] - max_epsilon, 0)
                            pertube_str[idx] = pertube_state[idx]

                    left = max(pertube_state[curr_indexes[0]] - max_epsilon, 0)
                    right = min(pertube_state[curr_indexes[0]] + max_epsilon, 1)
                    bottom = max(pertube_state[curr_indexes[1]] - max_epsilon, 0)
                    top = min(pertube_state[curr_indexes[1]] + max_epsilon, 1)

                    eps_x_range = np.linspace(start=left, stop=right, num=resolution, dtype=np.float32)
                    eps_y_range = np.linspace(start=bottom, stop=top, num=resolution, dtype=np.float32)
                    vals = [eps_x_range, eps_y_range]
                    eps_x, eps_y = np.meshgrid(eps_x_range, eps_y_range)

                    attacks_flag_matrix = vectorized_traj(eps_x, eps_y)
                    indexes_one_mask = np.nonzero(attacks_flag_matrix)  # return tuples if matrix
                    indexes_zero_mask = np.nonzero(1 - attacks_flag_matrix)  # return tuples  if matrix
                    if indexes_one_mask[0].size == 0:  # no attacks
                        rounded_min_epsilon_norm = f"No L_{order} attack found"
                    else:

                        if len(set(idx_states)) == 1:
                            points = np.array([[vals[0][x]] for x in set(indexes_one_mask[0])])
                            replica = np.full(shape=points.shape, fill_value=original_state[curr_indexes[0]],
                                              dtype=np.float32)
                        else:
                            points = np.array(
                                [[vals[0][x], vals[0][y]] for x, y in zip(indexes_one_mask[0], indexes_one_mask[1])])
                            replica_x = np.full(shape=([points.shape[0], 1]),
                                                fill_value=original_state[curr_indexes[0]],
                                                dtype=np.float32)
                            replica_y = np.full(shape=([points.shape[0], 1]),
                                                fill_value=original_state[curr_indexes[1]],
                                                dtype=np.float32)
                            replica = np.concatenate((replica_x, replica_y), axis=1)

                        min_epsilon_norm = np.min(np.linalg.norm(points - replica, ord=order, axis=1))
                        min_epsilon_norm_rounded = np.round(min_epsilon_norm, 3)
                        list_of_eps_per_dim.append(min_epsilon_norm_rounded)
                        rounded_min_epsilon_norm = f"min L_{order} eps:{min_epsilon_norm_rounded:.3f}"

                    # print(indexes_zero_mask)
                    if to_plot is True:
                        plt.scatter(np.choose(indexes_one_mask[0], vals[0]), np.choose(indexes_one_mask[1], vals[1]),
                                    color='r', s=10)
                        plt.scatter(np.choose(indexes_zero_mask[0], vals[0]), np.choose(indexes_zero_mask[1], vals[1]),
                                    color='b', s=10)
                        plt.scatter(pertube_state[curr_indexes[0]], pertube_state[curr_indexes[1]], color='k', s=50)
                        plt.text(pertube_state[curr_indexes[0]] + .02, pertube_state[curr_indexes[1]] + .02, real_label,
                                 fontsize=12)

                        # plt.imshow(abcd, extent=[left, right, bottom, top], cmap='jet')
                        plt.xlabel(str(curr_indexes[0]))
                        plt.ylabel(str(curr_indexes[1]))
                        if np.array_equal(original_state, pertube_state):  # in dim 1 or 2
                            plt.title(f'{pertube_str} \n {rounded_min_epsilon_norm}')
                        else:  # in dim 3 or more, where the rest are not changed
                            plt.title(f'{original_state} \n {pertube_str} \n {rounded_min_epsilon_norm}')
                        plt.show()
            min_eps = min(list_of_eps_per_dim)
            list_of_eps_dims.append(min_eps)
        return list_of_eps_dims
