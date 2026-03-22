import numpy as np


class RandomWalkSimulation:
    MODE_EXPLORATION = 0
    MODE_INSPECTION = 1
    MODE_EXPLOITATION = 2

    def __init__(
        self,
        num_dots: int,
        step_size: float,
        food_patches: list[dict],
        food_radius: float,
        agent_radius: float,
        fps: int,
        eat_cooldown_seconds: float,
        base_direction_reset_steps: int,
        min_direction_change_degrees: float,
        direction_noise_scale: float,
        levy_flight_probability: float,
        levy_flight_multiplier_min: float,
        levy_flight_multiplier_max: float,
        memory_size: int,
        memory_score_decay: float,
        memory_distance_merge: float,
        memory_revisit_strength: float,
        initial_patch_radius: float,
        min_patch_radius: float,
        max_patch_radius: float,
        patch_timeout_steps: int,
        patch_timeout_growth: int,
        patch_radius_shrink_factor: float,
        patch_radius_growth_factor: float,
        patch_confidence_gain: float,
        patch_confidence_decay: float,
        forage_pull_strength: float,
        forage_tangent_strength: float,
        forage_direction_blend: float,
        local_wander_scale: float,
        forage_speed_reduction: float,
        scent_radius: float,
        scent_strength: float,
        inspection_speed_reduction: float,
        social_signal_radius: float,
        social_signal_strength: float,
        social_signal_lifetime: int,
        use_boundary_reflection: bool,
        plot_limit: float,
    ):
        self.num_dots = num_dots
        self.step_size = step_size
        self.food_patches = food_patches
        self.food_count = sum(patch["count"] for patch in food_patches)
        self.food_radius = food_radius
        self.agent_radius = agent_radius
        self.fps = fps
        self.eat_cooldown_steps = max(1, int(round(eat_cooldown_seconds * fps)))

        self.base_direction_reset_steps = base_direction_reset_steps
        self.min_direction_change_radians = np.deg2rad(min_direction_change_degrees)
        self.direction_noise_scale = direction_noise_scale
        self.levy_flight_probability = levy_flight_probability
        self.levy_flight_multiplier_min = levy_flight_multiplier_min
        self.levy_flight_multiplier_max = levy_flight_multiplier_max

        self.memory_size = memory_size
        self.memory_score_decay = memory_score_decay
        self.memory_distance_merge = memory_distance_merge
        self.memory_revisit_strength = memory_revisit_strength

        self.initial_patch_radius = initial_patch_radius
        self.min_patch_radius = min_patch_radius
        self.max_patch_radius = max_patch_radius
        self.patch_timeout_steps = patch_timeout_steps
        self.patch_timeout_growth = patch_timeout_growth
        self.patch_radius_shrink_factor = patch_radius_shrink_factor
        self.patch_radius_growth_factor = patch_radius_growth_factor
        self.patch_confidence_gain = patch_confidence_gain
        self.patch_confidence_decay = patch_confidence_decay

        self.forage_pull_strength = forage_pull_strength
        self.forage_tangent_strength = forage_tangent_strength
        self.forage_direction_blend = forage_direction_blend
        self.local_wander_scale = local_wander_scale
        self.forage_speed_reduction = forage_speed_reduction

        self.scent_radius = scent_radius
        self.scent_strength = scent_strength
        self.inspection_speed_reduction = inspection_speed_reduction

        self.social_signal_radius = social_signal_radius
        self.social_signal_strength = social_signal_strength
        self.social_signal_lifetime = social_signal_lifetime

        self.use_boundary_reflection = use_boundary_reflection
        self.plot_limit = plot_limit

        self.positions = np.zeros((num_dots, 2), dtype=float)
        self.positions += np.random.normal(loc=0.0, scale=2.5, size=(num_dots, 2))

        self.search_directions = self._random_unit_vectors(num_dots)
        self.steps_since_food = np.zeros(num_dots, dtype=int)

        self.base_step_sizes = step_size * np.random.uniform(0.85, 1.15, size=num_dots)
        self.direction_reset_steps = np.random.randint(
            max(20, base_direction_reset_steps - 15),
            base_direction_reset_steps + 16,
            size=num_dots,
        )
        self.direction_persistence = np.random.uniform(0.55, 0.90, size=num_dots)

        self.cooldowns = np.zeros(num_dots, dtype=int)
        self.modes = np.full(num_dots, self.MODE_EXPLORATION, dtype=int)
        self.just_ate = np.zeros(num_dots, dtype=bool)

        self.has_patch_target = np.zeros(num_dots, dtype=bool)
        self.patch_centres = np.zeros((num_dots, 2), dtype=float)
        self.patch_radii = np.full(num_dots, initial_patch_radius, dtype=float)
        self.patch_confidence = np.zeros(num_dots, dtype=float)
        self.patch_steps_without_food = np.zeros(num_dots, dtype=int)
        self.patch_timeouts = np.full(num_dots, patch_timeout_steps, dtype=int)

        self.memory_centres = np.zeros((num_dots, memory_size, 2), dtype=float)
        self.memory_scores = np.zeros((num_dots, memory_size), dtype=float)
        self.memory_ages = np.zeros((num_dots, memory_size), dtype=int)
        self.memory_active = np.zeros((num_dots, memory_size), dtype=bool)

        self.food_positions = self._generate_food_positions()
        self.food_alive = np.ones(self.food_count, dtype=bool)

        self.signal_positions = np.empty((0, 2), dtype=float)
        self.signal_strengths = np.empty((0,), dtype=float)
        self.signal_ttls = np.empty((0,), dtype=int)

        self.history = [self.positions.copy()]
        self.food_history = [(self.food_positions.copy(), self.food_alive.copy())]

        self.total_food_eaten = 0
        self.current_step = 0

    def _generate_food_positions(self):
        all_food = []

        for patch in self.food_patches:
            center = np.array(patch["center"], dtype=float)
            spread = float(patch["spread"])
            count = int(patch["count"])

            patch_food = np.random.normal(
                loc=center,
                scale=spread,
                size=(count, 2),
            )
            all_food.append(patch_food)

        if not all_food:
            return np.empty((0, 2), dtype=float)

        return np.vstack(all_food)

    def _random_unit_vectors(self, n: int):
        angles = np.random.uniform(0, 2 * np.pi, n)
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def _normalise_vectors(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, np.maximum(norms, 1e-8))

    def _random_directions_with_min_turn(self, current_dirs, min_angle_radians):
        n = current_dirs.shape[0]
        new_dirs = np.zeros_like(current_dirs)
        filled = np.zeros(n, dtype=bool)

        while not np.all(filled):
            remaining = np.where(~filled)[0]
            candidates = self._random_unit_vectors(len(remaining))

            dots = np.sum(current_dirs[remaining] * candidates, axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)

            accept = angles >= min_angle_radians
            accepted_idx = remaining[accept]

            if accepted_idx.size > 0:
                new_dirs[accepted_idx] = candidates[accept]
                filled[accepted_idx] = True

        return new_dirs

    def _decay_memories(self):
        self.memory_scores *= self.memory_score_decay
        self.memory_ages[self.memory_active] += 1

        weak_mask = self.memory_active & (self.memory_scores < 0.05)
        self.memory_active[weak_mask] = False
        self.memory_scores[weak_mask] = 0.0
        self.memory_ages[weak_mask] = 0

    def _remember_food_patch(self, sheep_idx: int, food_pos: np.ndarray):
        active = self.memory_active[sheep_idx]

        if np.any(active):
            centres = self.memory_centres[sheep_idx, active]
            dists = np.linalg.norm(centres - food_pos, axis=1)
            nearest_local = np.argmin(dists)
            if dists[nearest_local] <= self.memory_distance_merge:
                memory_indices = np.where(active)[0]
                m_idx = memory_indices[nearest_local]
                old_score = self.memory_scores[sheep_idx, m_idx]
                self.memory_scores[sheep_idx, m_idx] = old_score + 1.0
                self.memory_centres[sheep_idx, m_idx] = (
                    0.7 * self.memory_centres[sheep_idx, m_idx] + 0.3 * food_pos
                )
                self.memory_ages[sheep_idx, m_idx] = 0
                return

        empty_slots = np.where(~self.memory_active[sheep_idx])[0]
        if empty_slots.size > 0:
            slot = empty_slots[0]
        else:
            slot = int(np.argmin(self.memory_scores[sheep_idx]))

        self.memory_active[sheep_idx, slot] = True
        self.memory_centres[sheep_idx, slot] = food_pos
        self.memory_scores[sheep_idx, slot] = 1.0
        self.memory_ages[sheep_idx, slot] = 0

    def _penalise_patch_memory(self, sheep_idx: int, patch_pos: np.ndarray):
        active = self.memory_active[sheep_idx]
        if not np.any(active):
            return

        centres = self.memory_centres[sheep_idx, active]
        dists = np.linalg.norm(centres - patch_pos, axis=1)
        nearest_local = np.argmin(dists)
        if dists[nearest_local] <= self.memory_distance_merge:
            memory_indices = np.where(active)[0]
            m_idx = memory_indices[nearest_local]
            self.memory_scores[sheep_idx, m_idx] *= 0.6

    def _best_memory_bias(self):
        bias = np.zeros((self.num_dots, 2), dtype=float)
        usable = np.any(self.memory_active, axis=1)

        if not np.any(usable):
            return bias

        for i in np.where(usable)[0]:
            active = self.memory_active[i]
            active_indices = np.where(active)[0]
            best_local = active_indices[np.argmax(self.memory_scores[i, active])]
            score = self.memory_scores[i, best_local]
            if score > 0:
                direction = self.memory_centres[i, best_local] - self.positions[i]
                direction = self._normalise_vectors(direction.reshape(1, 2))[0]
                bias[i] = score * direction

        return bias

    def _nearest_food_scent_bias(self):
        bias = np.zeros((self.num_dots, 2), dtype=float)
        scent_detected = np.zeros(self.num_dots, dtype=bool)

        alive_food = self.food_positions[self.food_alive]
        if alive_food.shape[0] == 0:
            return bias, scent_detected

        deltas = alive_food[None, :, :] - self.positions[:, None, :]
        dists = np.linalg.norm(deltas, axis=2)

        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(self.num_dots), nearest_idx]

        detected = nearest_dist <= self.scent_radius
        scent_detected[detected] = True

        if np.any(detected):
            nearest_food = alive_food[nearest_idx[detected]]
            vec = nearest_food - self.positions[detected]
            vec = self._normalise_vectors(vec)
            strength = (1.0 - (nearest_dist[detected] / self.scent_radius)).reshape(-1, 1)
            bias[detected] = strength * vec

        return bias, scent_detected

    def _social_bias(self):
        bias = np.zeros((self.num_dots, 2), dtype=float)
        if self.signal_positions.shape[0] == 0:
            return bias

        deltas = self.signal_positions[None, :, :] - self.positions[:, None, :]
        dists = np.linalg.norm(deltas, axis=2)

        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(self.num_dots), nearest_idx]

        detected = nearest_dist <= self.social_signal_radius
        if np.any(detected):
            nearest_signal = self.signal_positions[nearest_idx[detected]]
            vec = nearest_signal - self.positions[detected]
            vec = self._normalise_vectors(vec)

            base_strength = self.signal_strengths[nearest_idx[detected]].reshape(-1, 1)
            dist_strength = (1.0 - (nearest_dist[detected] / self.social_signal_radius)).reshape(-1, 1)
            bias[detected] = base_strength * dist_strength * vec

        return bias

    def _update_search_directions(self):
        reset_mask = (~self.has_patch_target) & (
            self.steps_since_food >= self.direction_reset_steps
        )

        reset_idx = np.where(reset_mask)[0]
        if reset_idx.size > 0:
            old_dirs = self.search_directions[reset_idx]
            new_dirs = self._random_directions_with_min_turn(
                old_dirs,
                self.min_direction_change_radians,
            )

            self.search_directions[reset_idx] = new_dirs
            self.steps_since_food[reset_idx] = 0

    def _update_patch_targets(self):
        active = np.where(self.has_patch_target)[0]
        if active.size == 0:
            return

        self.patch_confidence[active] *= self.patch_confidence_decay

        timed_out = active[
            self.patch_steps_without_food[active] >= self.patch_timeouts[active]
        ]

        if timed_out.size > 0:
            for idx in timed_out:
                self._penalise_patch_memory(idx, self.patch_centres[idx])
            self.has_patch_target[timed_out] = False
            self.patch_steps_without_food[timed_out] = 0
            self.patch_confidence[timed_out] = 0.0
            self.patch_radii[timed_out] = self.initial_patch_radius
            self.patch_timeouts[timed_out] = self.patch_timeout_steps
            self.search_directions[timed_out] = self._random_unit_vectors(timed_out.size)

    def _update_signals(self):
        if self.signal_positions.shape[0] == 0:
            return

        self.signal_ttls -= 1
        keep = self.signal_ttls > 0

        self.signal_positions = self.signal_positions[keep]
        self.signal_strengths = self.signal_strengths[keep]
        self.signal_ttls = self.signal_ttls[keep]

    def _add_social_signal(self, pos: np.ndarray):
        self.signal_positions = np.vstack([self.signal_positions, pos.reshape(1, 2)])
        self.signal_strengths = np.append(self.signal_strengths, 1.0)
        self.signal_ttls = np.append(self.signal_ttls, self.social_signal_lifetime)

    def _apply_boundary_reflection(self):
        if not self.use_boundary_reflection:
            return

        x_low = self.positions[:, 0] < -self.plot_limit
        x_high = self.positions[:, 0] > self.plot_limit
        y_low = self.positions[:, 1] < -self.plot_limit
        y_high = self.positions[:, 1] > self.plot_limit

        self.positions[x_low, 0] = -self.plot_limit
        self.positions[x_high, 0] = self.plot_limit
        self.positions[y_low, 1] = -self.plot_limit
        self.positions[y_high, 1] = self.plot_limit

        self.search_directions[x_low | x_high, 0] *= -1
        self.search_directions[y_low | y_high, 1] *= -1
        self.search_directions = self._normalise_vectors(self.search_directions)

    def _move_agents(self):
        self.just_ate[:] = False

        self._decay_memories()
        self._update_signals()
        self._update_patch_targets()
        self._update_search_directions()

        random_noise = self._random_unit_vectors(self.num_dots) * self.direction_noise_scale
        base_dirs = self._normalise_vectors(self.search_directions + random_noise)

        memory_bias = self._best_memory_bias()
        memory_bias = self._normalise_vectors(memory_bias + 1e-12) * (
            np.linalg.norm(memory_bias, axis=1, keepdims=True) > 1e-6
        )

        scent_bias, scent_detected = self._nearest_food_scent_bias()
        social_bias = self._social_bias()

        move_dirs = base_dirs.copy()
        step_scales = self.base_step_sizes.copy()

        self.modes[:] = self.MODE_EXPLORATION
        inspect_mask = (~self.has_patch_target) & (
            scent_detected | (np.linalg.norm(social_bias, axis=1) > 0)
        )
        self.modes[inspect_mask] = self.MODE_INSPECTION
        self.modes[self.has_patch_target] = self.MODE_EXPLOITATION

        exploration_idx = np.where(self.modes == self.MODE_EXPLORATION)[0]
        if exploration_idx.size > 0:
            combined = (
                0.62 * base_dirs[exploration_idx]
                + self.memory_revisit_strength * memory_bias[exploration_idx]
                + 0.10 * social_bias[exploration_idx]
                + 0.06 * scent_bias[exploration_idx]
                + self._random_unit_vectors(exploration_idx.size) * 0.15
            )
            move_dirs[exploration_idx] = self._normalise_vectors(combined)

            levy_mask = np.random.rand(exploration_idx.size) < self.levy_flight_probability
            if np.any(levy_mask):
                long_steps = np.random.uniform(
                    self.levy_flight_multiplier_min,
                    self.levy_flight_multiplier_max,
                    size=np.sum(levy_mask),
                )
                step_scales[exploration_idx[levy_mask]] *= long_steps

        inspection_idx = np.where(self.modes == self.MODE_INSPECTION)[0]
        if inspection_idx.size > 0:
            combined = (
                0.32 * base_dirs[inspection_idx]
                + self.scent_strength * scent_bias[inspection_idx]
                + self.social_signal_strength * social_bias[inspection_idx]
                + 0.18 * memory_bias[inspection_idx]
                + self._random_unit_vectors(inspection_idx.size) * 0.18
            )
            move_dirs[inspection_idx] = self._normalise_vectors(combined)
            step_scales[inspection_idx] *= self.inspection_speed_reduction

        exploitation_idx = np.where(self.modes == self.MODE_EXPLOITATION)[0]
        if exploitation_idx.size > 0:
            to_patch = self.patch_centres[exploitation_idx] - self.positions[exploitation_idx]
            dist = np.linalg.norm(to_patch, axis=1)
            to_patch_unit = self._normalise_vectors(to_patch)
            tangential = np.column_stack((-to_patch_unit[:, 1], to_patch_unit[:, 0]))

            local_scent = scent_bias[exploitation_idx]
            local_noise = self._random_unit_vectors(exploitation_idx.size) * self.local_wander_scale

            combined = (
                self.forage_pull_strength * to_patch_unit
                + self.forage_tangent_strength * tangential
                + self.forage_direction_blend * base_dirs[exploitation_idx]
                + 0.22 * local_scent
                + local_noise
            )
            combined = self._normalise_vectors(combined)

            near_mask = dist < (0.35 * self.patch_radii[exploitation_idx])
            if np.any(near_mask):
                near = np.where(near_mask)[0]
                near_combined = (
                    0.72 * tangential[near]
                    + 0.12 * base_dirs[exploitation_idx][near]
                    + self._random_unit_vectors(len(near)) * (self.local_wander_scale + 0.08)
                )
                combined[near] = self._normalise_vectors(near_combined)

            far_mask = dist > self.patch_radii[exploitation_idx]
            if np.any(far_mask):
                far = np.where(far_mask)[0]
                far_combined = (
                    0.78 * to_patch_unit[far]
                    + 0.12 * tangential[far]
                    + self._random_unit_vectors(len(far)) * 0.10
                )
                combined[far] = self._normalise_vectors(far_combined)

            move_dirs[exploitation_idx] = combined

            confidence_scale = 1.0 / (1.0 + 0.12 * self.patch_confidence[exploitation_idx])
            step_scales[exploitation_idx] *= self.forage_speed_reduction * confidence_scale

        self.positions += step_scales.reshape(-1, 1) * move_dirs
        self.steps_since_food += 1
        self.patch_steps_without_food[self.has_patch_target] += 1

        self._apply_boundary_reflection()

    def _handle_food_collisions(self):
        alive_indices = np.where(self.food_alive)[0]
        if alive_indices.size == 0:
            return

        active_food = self.food_positions[alive_indices]

        for sheep_idx in range(self.num_dots):
            if self.cooldowns[sheep_idx] > 0:
                continue

            deltas = active_food - self.positions[sheep_idx]
            dists = np.linalg.norm(deltas, axis=1)
            hit_mask = dists <= (self.food_radius + self.agent_radius)

            if not np.any(hit_mask):
                continue

            nearest_local = np.argmin(np.where(hit_mask, dists, np.inf))
            food_global_idx = alive_indices[nearest_local]
            eaten_food_pos = self.food_positions[food_global_idx].copy()

            self.food_alive[food_global_idx] = False
            self.cooldowns[sheep_idx] = self.eat_cooldown_steps
            self.just_ate[sheep_idx] = True
            self.total_food_eaten += 1

            self._remember_food_patch(sheep_idx, eaten_food_pos)

            if self.has_patch_target[sheep_idx]:
                same_patch = (
                    np.linalg.norm(self.patch_centres[sheep_idx] - eaten_food_pos)
                    <= self.patch_radii[sheep_idx]
                )
            else:
                same_patch = False

            self.has_patch_target[sheep_idx] = True
            self.patch_centres[sheep_idx] = eaten_food_pos
            self.patch_steps_without_food[sheep_idx] = 0

            if same_patch:
                self.patch_confidence[sheep_idx] += self.patch_confidence_gain
                self.patch_radii[sheep_idx] = max(
                    self.min_patch_radius,
                    self.patch_radii[sheep_idx] * self.patch_radius_shrink_factor,
                )
                self.patch_timeouts[sheep_idx] += self.patch_timeout_growth
            else:
                self.patch_confidence[sheep_idx] = 1.0
                self.patch_radii[sheep_idx] = self.initial_patch_radius
                self.patch_timeouts[sheep_idx] = self.patch_timeout_steps

            self.patch_radii[sheep_idx] = min(
                self.max_patch_radius, self.patch_radii[sheep_idx]
            )

            self.steps_since_food[sheep_idx] = 0
            self.search_directions[sheep_idx] = self._random_unit_vectors(1)[0]

            self._add_social_signal(eaten_food_pos)

            alive_indices = np.where(self.food_alive)[0]
            active_food = self.food_positions[alive_indices]
            if alive_indices.size == 0:
                break

        exploiters = np.where(self.has_patch_target)[0]
        if exploiters.size > 0:
            widen_mask = self.patch_steps_without_food[exploiters] > 8
            widen_idx = exploiters[widen_mask]
            if widen_idx.size > 0:
                self.patch_radii[widen_idx] = np.minimum(
                    self.max_patch_radius,
                    self.patch_radii[widen_idx] * self.patch_radius_growth_factor,
                )

    def _decay_states(self):
        self.cooldowns = np.maximum(self.cooldowns - 1, 0)

    def get_state_counts(self):
        return {
            "exploring": int(np.sum(self.modes == self.MODE_EXPLORATION)),
            "inspecting": int(np.sum(self.modes == self.MODE_INSPECTION)),
            "exploiting": int(np.sum(self.modes == self.MODE_EXPLOITATION)),
            "cooldown": int(np.sum(self.cooldowns > 0)),
            "food_remaining": int(np.sum(self.food_alive)),
            "food_eaten": int(self.total_food_eaten),
        }

    def step(self):
        self.current_step += 1
        self._move_agents()
        self._handle_food_collisions()

        self.history.append(self.positions.copy())
        self.food_history.append((self.food_positions.copy(), self.food_alive.copy()))

        self._decay_states()
        return self.positions, self.food_positions, self.food_alive