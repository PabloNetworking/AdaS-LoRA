"""
FAH-QLoRA Rank Scheduler for Federated Adaptive Heterogeneous LoRA.

This module implements the two-stage rank selection algorithm from FAH-QLoRA:
  Stage 1: Adapt average LoRA rank r_i across rounds by maximizing loss decrease rate
  Stage 2: Per-device rank assignment to minimize round completion time under constraints

Time Modeling (equations 12-14):
  - Computation time: t_cmp^n(r) = alpha_n + (r / r_max) * t_lora_n
  - Communication time: t_com^n(r) = L0 / b_dn_n + L(r) / b_up_n
  - Round time: T = max_n(t_cmp^n + t_com^n)

Per-device bandwidth comes from the shared bandwidth manager and is refreshed every round;
the computation part of T_i uses the clients' measured training time.

Reference: FAH-QLoRA paper equations (7)-(17) and Algorithm 1.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class FahRankScheduler:
    """
    Scheduler for FAH-QLoRA dynamic rank adaptation.
    
    Maintains history of ranks, losses, and times, and provides methods for:
    - Registering client profiles (compute time params + bandwidth)
    - Updating round statistics (losses from clients, times computed analytically)
    - Computing average rank updates (Stage 1 - gradient sign approximation)
    - Solving per-device rank assignment (Stage 2 - P1 optimization)
    
    Key design principles:
    - Per-device bandwidth is refreshed from the shared bandwidth manager every round
    - Communication time and the hat-rank time T̂_i are computed analytically using
      _compute_client_time; T_i can use measured client compute times (time_pairs)
    - Profiling happens during warmup to estimate alpha_n and t_lora_n
    """
    
    def __init__(
        self,
        init_rank: int = 8,
        r_min: int = 2,
        r_max: int = 64,
        lambda_dec: float = 1.0,
        lambda_inc: float = 1.0,
        warmup_rounds: int = 2,
        client_num: int = 1,
        alpha_fraction: float = 0.3,
        client_rank_caps: Optional[Dict[int, int]] = None,
    ):
        """
        Initialize the FAH rank scheduler.
        
        Args:
            init_rank: Initial homogeneous rank r_0 for warmup
            r_min: Minimum allowed rank
            r_max: Maximum allowed rank (also used in time model)
            lambda_dec: Step size λ1 for rank decrease (eq. 11)
            lambda_inc: Step size λ2 for rank increase (eq. 11)
            warmup_rounds: Number of homogeneous warm-up rounds before FAH activates
            client_num: Expected number of clients
            client_rank_caps: Optional immutable per-client scalar rank caps
        """
        self.init_rank = init_rank
        self.r_min = r_min
        self.r_max = r_max
        self.lambda_dec = lambda_dec
        self.lambda_inc = lambda_inc
        self.warmup_rounds = warmup_rounds
        self.client_num = client_num
        self.alpha_fraction = alpha_fraction
        self.client_rank_caps: Dict[int, int] = self._normalize_client_rank_caps(client_rank_caps)
        
        # History tracking
        self.r_history: List[float] = [float(init_rank)]  # Average ranks r_i
        self.r_hat_history: List[float] = [max(r_min, init_rank - 1)]  # "Hat" ranks r̂_i
        self.F_history: List[float] = []  # Global losses F_i (at current rank)
        self.F_hat_history: List[float] = []  # Global losses F̂_i (at hat rank)
        self.T_history: List[float] = []  # Round completion times T_i
        self.T_hat_history: List[float] = []  # Times at hat rank T̂_i
        self.R_history: List[float] = []  # Loss decrease rates R_i
        self.R_hat_history: List[float] = []  # Loss decrease rates R̂_i at hat rank
        
        # Per-client profiles: alpha and t_lora are set after warmup; b_up and b_dn are
        # refreshed every round
        # Format: {client_id: {'alpha': α_n, 't_lora': t_lora_n, 'b_up': b_up_n, 'b_dn': b_dn_n}}
        self.client_profiles: Dict[int, Dict[str, float]] = {}
        
        # LoRA size parameters for communication time (set once after model creation)
        self.L0_bytes: float = 0.0  # Size of global LoRA at max_rank (bytes)
        self.unit_lora_bytes: float = 0.0  # Size per unit rank (bytes), L(r) = unit_lora_bytes * r
        
        # Current round's per-client rank assignments
        self.current_ranks: Dict[int, int] = {}  # client_id -> r_i^n (training rank)
        self.current_hat_ranks: Dict[int, int] = {}  # client_id -> r̂_i^n (evaluation rank)
        
        logger.info(
            f"Initialized FahRankScheduler: init_rank={init_rank}, "
            f"r_range=[{r_min}, {r_max}], λ_dec={lambda_dec}, λ_inc={lambda_inc}, "
            f"warmup={warmup_rounds} rounds, external_caps={len(self.client_rank_caps)}"
        )

    def _normalize_client_rank_caps(self, client_rank_caps: Optional[Dict[int, int]]) -> Dict[int, int]:
        normalized = {}
        if not client_rank_caps:
            return normalized

        for raw_client_id, raw_cap in client_rank_caps.items():
            client_id = raw_client_id
            if isinstance(raw_client_id, str) and raw_client_id.startswith('Client_'):
                client_id = raw_client_id.split('_')[-1]

            try:
                client_id = int(client_id)
                cap = int(raw_cap)
            except (TypeError, ValueError):
                continue

            normalized[client_id] = cap

        return normalized

    def _get_client_rank_cap(self, client_id: int) -> int:
        cap = self.client_rank_caps.get(client_id, self.r_max)
        return max(self.r_min, min(self.r_max, int(cap)))

    def _get_feasible_stage2_target(
        self,
        client_ids: List[int],
        r_i: float,
    ) -> Tuple[int, int, float]:
        target_sum = int(round(len(client_ids) * r_i))
        min_sum = len(client_ids) * self.r_min
        max_sum = sum(self._get_client_rank_cap(cid) for cid in client_ids)
        feasible_target_sum = max(min_sum, min(target_sum, max_sum))
        feasible_target_avg = (
            feasible_target_sum / float(len(client_ids))
            if client_ids else float(r_i)
        )
        return target_sum, feasible_target_sum, feasible_target_avg

    def register_client(
        self,
        client_id: int,
        alpha_n: float,
        t_lora_n: float,
        b_up_n: float,
        b_dn_n: float,
        b_up_units: str = 'Mbps',
        b_dn_units: str = 'Mbps',
    ) -> None:
        """
        Register a client's profile for analytical time estimation.
        
        Call this once per client after warmup profiling; bandwidth is refreshed every round.
        
        Args:
            client_id: Client identifier
            alpha_n: Base computation time (time without LoRA, or can be 0)
            t_lora_n: LoRA computation time at r_max (scaled linearly with rank)
            b_up_n: Uplink bandwidth value
            b_dn_n: Downlink bandwidth value
            b_up_units: Units for uplink bandwidth ('Mbps' or 'kbit/s')
            b_dn_units: Units for downlink bandwidth ('Mbps' or 'kbit/s')
        """
        self.client_profiles[client_id] = {
            'alpha': alpha_n,
            't_lora': t_lora_n,
            'b_up': b_up_n,
            'b_dn': b_dn_n,
            'b_up_units': b_up_units,
            'b_dn_units': b_dn_units,
        }
        
        # Initialize ranks to init_rank
        cap = self._get_client_rank_cap(client_id)
        init_rank_capped = min(self.init_rank, cap)

        self.current_ranks[client_id] = init_rank_capped
        self.current_hat_ranks[client_id] = max(self.r_min, min(cap, init_rank_capped - 1))
        
        logger.info(
            f"Registered client {client_id}: α={alpha_n:.3f}s, "
            f"t_lora={t_lora_n:.3f}s, b_up={b_up_n:.2f}{b_up_units}, b_dn={b_dn_n:.2f}{b_dn_units}"
        )
    
    def set_lora_size(self, L0_bytes: float, unit_lora_bytes: float) -> None:
        """
        Set LoRA size parameters for communication time estimation.
        
        Call this once after the model is created.
        
        Args:
            L0_bytes: Total size of LoRA parameters at max_rank (bytes)
            unit_lora_bytes: Size per unit rank (bytes), so L(r) = unit_lora_bytes * r
        """
        self.L0_bytes = L0_bytes
        self.unit_lora_bytes = unit_lora_bytes
        logger.info(
            f"Set LoRA size: L0={L0_bytes/1e6:.2f}MB (at r_max), "
            f"per_rank={unit_lora_bytes/1e3:.2f}KB"
        )
    
    def _compute_client_time(
        self,
        client_id: int,
        rank: int,
    ) -> Tuple[float, float]:
        """
        Compute computation and communication time for a client at given rank.
        
        Implements equations (12)-(14):
            t_cmp^n(r) = alpha_n + (r / r_max) * t_lora_n
            t_com^n(r) = L0 / b_dn_n + L(r) / b_up_n
        
        where L(r) = unit_lora_bytes * r
        
        Args:
            client_id: Client identifier
            rank: LoRA rank to compute time for
            
        Returns:
            Tuple of (t_cmp, t_com) - computation time and communication time in seconds
        """
        if client_id not in self.client_profiles:
            logger.warning(f"Client {client_id} not registered, using defaults")
            return 1.0, 1.0
        
        profile = self.client_profiles[client_id]
        alpha_n = profile['alpha']
        t_lora_n = profile['t_lora']
        b_up_n = profile['b_up']
        b_dn_n = profile['b_dn']
        b_up_units = profile.get('b_up_units', 'Mbps')
        b_dn_units = profile.get('b_dn_units', 'Mbps')
        
        # Equation (12): Computation time scales linearly with rank
        t_cmp = alpha_n + (rank / self.r_max) * t_lora_n
        
        # Equation (13)-(14): Communication time
        # Convert bandwidth to bytes/s based on units
        # If in kbit/s: convert to Mbps first (divide by 1000), then to bytes/s
        # If in Mbps: convert directly to bytes/s
        # 1 Mbps = 1,000,000 bits/s = 125,000 bytes/s
        # 1 kbit/s = 1,000 bits/s = 125 bytes/s
        
        if b_up_units == 'kbit/s':
            # Convert kbit/s to bytes/s: 1 kbit/s = 1,000 bits/s = 125 bytes/s
            b_up_bytes = b_up_n * 125
        else:  # Mbps (default or explicit)
            # Convert Mbps to bytes/s: 1 Mbps = 1,000,000 bits/s = 125,000 bytes/s
            b_up_bytes = b_up_n * 125000
        
        if b_dn_units == 'kbit/s':
            # Convert kbit/s to bytes/s: 1 kbit/s = 1,000 bits/s = 125 bytes/s
            b_dn_bytes = b_dn_n * 125
        else:  # Mbps (default or explicit)
            # Convert Mbps to bytes/s: 1 Mbps = 1,000,000 bits/s = 125,000 bytes/s
            b_dn_bytes = b_dn_n * 125000
        
        
        # Download: receive global LoRA at max_rank (L0)
        # Upload: send local LoRA at client's rank (L(r))
        L_r = self.unit_lora_bytes * rank
        t_com = self.L0_bytes / b_dn_bytes + L_r / b_up_bytes
        
        return t_cmp, t_com
    
    def update_round_stats(
        self,
        round_idx: int,
        local_loss_pairs: Dict[int, Tuple[float, float]],
        time_pairs: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        previous_global_loss: Optional[float] = None,
    ) -> None:
        """
        Update round statistics from client feedback.
        
        Two modes of operation:
        1. If time_pairs is None: All times (compute + comm) are computed analytically
           using _compute_client_time with stored client profiles.
        2. If time_pairs is provided: Uses measured compute times from clients for T_i,
           but keeps communication time analytic. T_hat_i remains fully analytic
           (counterfactual quantity).
        
        Args:
            round_idx: Current round index
            local_loss_pairs: Dict {client_id: (F_rank, F_hat_rank)}
                - F_rank: validation loss at current rank r_i^n
                - F_hat_rank: validation loss at hat rank r̂_i^n
            time_pairs: (Optional) Dict {client_id: (measured_t_cmp, _, _, _)}
                If provided, the first element (measured compute time) is used for T_i,
                while communication time is computed analytically. The other elements
                are ignored; T_hat_i is computed fully analytically as a counterfactual.
                If None (default), all times are computed analytically.
            previous_global_loss: F_{i-1}, the global loss from the previous round.
                Used to compute loss decrease rate R_i = (F_{i-1} - F_i) / T_i
        """
        if not local_loss_pairs:
            logger.warning("update_round_stats called with empty loss pairs")
            return
        
        # Compute global average losses (simple mean across clients)
        F_values = [pair[0] for pair in local_loss_pairs.values()]
        F_hat_values = [pair[1] for pair in local_loss_pairs.values()]
        
        F_i = np.mean(F_values)
        F_hat_i = np.mean(F_hat_values)
        
        self.F_history.append(F_i)
        self.F_hat_history.append(F_hat_i)
        
        # Compute round completion times analytically using _compute_client_time,
        # unless time_pairs provides measured compute times
        if time_pairs is None:
            # Use analytical time model with stored profiles and current ranks
            T_values = []
            T_hat_values = []
            
            for client_id in local_loss_pairs.keys():
                rank = self.current_ranks.get(client_id, self.init_rank)
                hat_rank = self.current_hat_ranks.get(client_id, max(self.r_min, rank - 1))
                
                t_cmp, t_com = self._compute_client_time(client_id, rank)
                t_hat_cmp, t_hat_com = self._compute_client_time(client_id, hat_rank)
                
                T_values.append(t_cmp + t_com)
                T_hat_values.append(t_hat_cmp + t_hat_com)

                logger.info(
                    f"Client {client_id} stats: t_cmp={t_cmp:.4f}s, t_com={t_com:.4f}s, "
                    f"t_hat_cmp={t_hat_cmp:.4f}s, t_hat_com={t_hat_com:.4f}s."
                 )
            
            # Round time is straggler-dominated (max over clients)
            T_i = max(T_values) if T_values else 1.0
            T_hat_i = max(T_hat_values) if T_hat_values else 1.0
        else:
            # Use measured compute times but keep communication time analytic
            logger.debug("Using measured compute times (time_pairs) with analytic comm model")
            T_values = []
            T_hat_values = []
            
            for client_id, times in time_pairs.items():
                # Only use the measured compute time from time_pairs
                measured_cmp, _, _, _ = times
                
                # Get current and hat ranks as usual
                rank = self.current_ranks.get(client_id, self.init_rank)
                hat_rank = self.current_hat_ranks.get(client_id, max(self.r_min, rank - 1))
                
                # Compute analytic communication time (ignore analytic t_cmp for current rank)
                # and fully analytic hat-side compute+comm
                _, t_com = self._compute_client_time(client_id, rank)
                t_hat_cmp, t_hat_com = self._compute_client_time(client_id, hat_rank)
                
                # T_i: measured compute + analytic comm
                T_values.append(measured_cmp + t_com)
                
                # T_hat_i: keep fully analytic (counterfactual)
                T_hat_values.append(t_hat_cmp + t_hat_com)
                
                logger.info(
                    f"Client {client_id} stats: measured_cmp={measured_cmp:.4f}s, t_hat_cmp={t_hat_cmp:.4f}s, "
                    f"t_comm={t_com:.4f}s, t_hat_comm={t_hat_com:.4f}s."
                )
            
            T_i = max(T_values) if T_values else 1.0
            T_hat_i = max(T_hat_values) if T_hat_values else 1.0
        
        self.T_history.append(T_i)
        self.T_hat_history.append(T_hat_i)
        
        # Compute loss decrease rates (equation 17)
        # R_i = (F_{i-1} - F_i) / T_i
        # R̂_i = (F_{i-1} - F̂_i) / T̂_i
        if previous_global_loss is not None and T_i > 0 and T_hat_i > 0:
            R_i = (previous_global_loss - F_i) / T_i
            R_hat_i = (previous_global_loss - F_hat_i) / T_hat_i
            
            self.R_history.append(R_i)
            self.R_hat_history.append(R_hat_i)
            
            logger.info(
                f"Round {round_idx} stats: F={F_i:.4f}, F̂={F_hat_i:.4f}, "
                f"T={T_i:.2f}s, T̂={T_hat_i:.2f}s, R={R_i:.6f}, R̂={R_hat_i:.6f}"
            )
        else:
            logger.info(
                f"Round {round_idx} stats (warmup): F={F_i:.4f}, F̂={F_hat_i:.4f}, "
                f"T={T_i:.2f}s."
            )
    
    def update_average_rank(self, round_idx: int) -> Tuple[float, float]:
        """
        Update average rank using gradient sign approximation (Stage 1).
        
        Implements equations (10)-(11):
            sign(∇f(r_{i-1})) = sign((R̂_{i-1} - R_{i-1}) / (r_{i-1} - r̂_{i-1}))
            
            r_i = r_{i-1} - λ1  if sign > 0  (decrease rank)
            r_i = r_{i-1} + λ2  if sign < 0  (increase rank)
            r_i = r_{i-1}       if sign = 0
        
        Note: This method only updates r_history and r_hat_history.
        Per-client ranks (current_ranks, current_hat_ranks) are updated by solve_p1_for_round.
        
        Args:
            round_idx: Current round index
            
        Returns:
            Tuple of (r_i, r̂_i) - new average rank and average hat rank
        """
        if len(self.R_history) < 1 or len(self.R_hat_history) < 1:
            # Not enough data yet, keep current rank
            r_i = self.r_history[-1]
            r_hat_i = self.r_hat_history[-1]
            logger.info(f"Round {round_idx}: Insufficient R data, keeping r={r_i}")
            return r_i, r_hat_i
        
        # Get latest loss decrease rates
        R_last = self.R_history[-1]
        R_hat_last = self.R_hat_history[-1]
        
        # Get previous average ranks
        r_prev = self.r_history[-1]
        r_hat_prev = self.r_hat_history[-1]
        
        # Equation (10): Compute gradient sign
        # denominator = r_{i-1} - r̂_{i-1}: positive, since r̂ = r - 1 (zero once r reaches r_min)
        denominator = r_prev - r_hat_prev
        
        eps = 1e-8
        if abs(denominator) < eps:
            sign_gradient = 0
            logger.warning(f"Round {round_idx}: sign_gradient equals 0.")
        else:
            sign_gradient = np.sign((R_hat_last - R_last) / denominator)
        
        # Equation (11): Update rank based on gradient sign
        if sign_gradient > 0:
            # Positive gradient: decreasing rank improves loss decrease rate
            r_i = r_prev - self.lambda_dec
        elif sign_gradient < 0:
            # Negative gradient: increasing rank improves loss decrease rate
            r_i = r_prev + self.lambda_inc
        else:
            # Zero gradient: keep current rank
            r_i = r_prev
        
        # Clip to [r_min, r_max]
        r_i = float(max(self.r_min, min(self.r_max, r_i)))
        
        # Hat rank is slightly lower
        r_hat_i = float(max(self.r_min, r_i - 1))
        
        # Update history (but NOT current_ranks - that's done in solve_p1)
        self.r_history.append(r_i)
        self.r_hat_history.append(r_hat_i)
        
        logger.info(
            f"Round {round_idx} Stage 1: r={r_i:.1f} (prev={r_prev:.1f}), "
            f"sign={sign_gradient}, R={R_last:.6f}, R̂={R_hat_last:.6f}"
        )
        
        return r_i, r_hat_i
    
    def _redistribute_deficit_under_caps(
    self,
    clamped_r: Dict[int, int],
    client_ids: List[int],
    feasible_target_sum: int,
    ) -> Dict[int, int]:
        """
        Redistribute leftover rank units after hard-cap clamping so that the final
        sum of assigned ranks matches the feasible Stage 1 target whenever possible.

        Strategy:
        - Compute deficit = feasible_target_sum - current_sum
        - If deficit > 0, allocate extra rank units to clients that are still below
        their hard cap
        - Prefer faster clients first, using smaller B_n from _get_time_coefficients()
        as the priority signal, matching FAH's latency-aware intuition

        Args:
            clamped_r: Current per-client ranks after hard-cap clamp
            client_ids: Ordered list of active client IDs
            feasible_target_sum: Target total rank after feasibility clamp

        Returns:
            Updated per-client ranks after redistribution
        """
        current_sum = sum(clamped_r.values())
        deficit = feasible_target_sum - current_sum

        if deficit <= 0:
            return clamped_r

        # Rank clients by smaller B_n first (faster per-rank time cost)
        latency_order = []
        for cid in client_ids:
            _, B_n = self._get_time_coefficients(cid)
            latency_order.append((cid, B_n))
        latency_order.sort(key=lambda x: x[1])

        # Total extra capacity still available under hard caps
        spare_capacity = {
            cid: max(0, self._get_client_rank_cap(cid) - clamped_r[cid])
            for cid in client_ids
        }
        total_spare = sum(spare_capacity.values())

        if total_spare <= 0:
            logger.info(
                f"Redistribution skipped: no spare capacity under hard caps "
                f"(current_sum={current_sum}, target_sum={feasible_target_sum})"
            )
            return clamped_r

        # Cannot allocate more than available spare capacity
        deficit = min(deficit, total_spare)

        # Greedy one-unit redistribution to fastest clients first
        while deficit > 0:
            progressed = False
            for cid, _ in latency_order:
                if spare_capacity[cid] > 0 and deficit > 0:
                    clamped_r[cid] += 1
                    spare_capacity[cid] -= 1
                    deficit -= 1
                    progressed = True
                if deficit == 0:
                    break

            if not progressed:
                break

        return clamped_r
        
    def solve_p1_for_round(
    self,
    round_idx: int,
    r_i: float,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Solve P1 for round i to assign per-device ranks.

        Per-client rank caps are enforced as follows:
        1. Clamp the Stage 1 target to the feasible sum implied by immutable per-client caps
        2. Solve Stage 2 with per-client upper bounds
        3. Keep a post-solve safety clamp + redistribution as a final guardrail
        """
        client_ids = list(self.client_profiles.keys())
        N = len(client_ids)

        if N == 0:
            logger.warning("solve_p1_for_round: No clients registered")
            return {}, {}

        target_sum, feasible_target_sum, feasible_r_i = self._get_feasible_stage2_target(
            client_ids=client_ids,
            r_i=r_i,
        )

        if feasible_target_sum != target_sum:
            logger.info(
                f"Round {round_idx} Stage 2 target clamped by immutable caps: "
                f"requested_sum={target_sum}, feasible_sum={feasible_target_sum}, "
                f"requested_avg={r_i:.2f}, feasible_avg={feasible_r_i:.2f}"
            )

        try:
            r_per_client, r_hat_per_client = self._solve_p1_cvxpy(
                round_idx, feasible_r_i, client_ids, N, feasible_target_sum
            )
        except ImportError:
            logger.info("cvxpy not available, using heuristic solver")
            r_per_client, r_hat_per_client = self._solve_p1_heuristic(
                round_idx, feasible_r_i, client_ids, N, feasible_target_sum
            )
        except Exception as e:
            logger.warning(f"cvxpy failed: {e}, using heuristic")
            r_per_client, r_hat_per_client = self._solve_p1_heuristic(
                round_idx, feasible_r_i, client_ids, N, feasible_target_sum
            )

        clamped_r: Dict[int, int] = {}
        for client_id, rank in r_per_client.items():
            cap = self._get_client_rank_cap(client_id)
            rank_capped = max(self.r_min, min(cap, int(rank)))
            clamped_r[client_id] = rank_capped

        before_redistribution_sum = sum(clamped_r.values())
        clamped_r = self._redistribute_deficit_under_caps(
            clamped_r=clamped_r,
            client_ids=client_ids,
            feasible_target_sum=feasible_target_sum,
        )
        after_redistribution_sum = sum(clamped_r.values())

        clamped_r_hat: Dict[int, int] = {}
        for client_id, rank in clamped_r.items():
            cap = self._get_client_rank_cap(client_id)
            clamped_r_hat[client_id] = max(self.r_min, min(cap, rank - 1))

        self.current_ranks = clamped_r.copy()
        self.current_hat_ranks = clamped_r_hat.copy()

        logger.info(
            f"Round {round_idx} Stage 2 final: "
            f"requested_avg={r_i:.2f}, "
            f"feasible_avg={feasible_r_i:.2f}, "
            f"requested_sum={target_sum}, "
            f"feasible_target_sum={feasible_target_sum}, "
            f"sum_after_solver={before_redistribution_sum}, "
            f"sum_final={after_redistribution_sum}, "
            f"ranks={list(clamped_r.values())}, "
            f"caps={[self._get_client_rank_cap(cid) for cid in sorted(clamped_r.keys())]}"
        )

        return clamped_r, clamped_r_hat

    def _get_time_coefficients(self, client_id: int) -> Tuple[float, float]:
        """
        Get time coefficients A_n and B_n for P1 optimization.
        
        T_n(r) = A_n + B_n * r where:
            A_n = alpha_n + L0 / b_dn_n  (base time independent of rank)
            B_n = t_lora_n / r_max + unit_lora_bytes / b_up_n  (per-rank time)
        
        Returns:
            Tuple (A_n, B_n)
        """
        profile = self.client_profiles.get(client_id, {})
        alpha = profile.get('alpha', 0.0)
        t_lora = profile.get('t_lora', 1.0)
        b_up_val = profile.get('b_up', 10.0)
        b_dn_val = profile.get('b_dn', 50.0)
        b_up_units = profile.get('b_up_units', 'Mbps')
        b_dn_units = profile.get('b_dn_units', 'Mbps')
        
        # Convert to bytes/s based on units
        if b_up_units == 'kbit/s':
            b_up = b_up_val * 125  # kbit/s to bytes/s
        else:
            b_up = b_up_val * 125000  # Mbps to bytes/s
        
        if b_dn_units == 'kbit/s':
            b_dn = b_dn_val * 125  # kbit/s to bytes/s
        else:
            b_dn = b_dn_val * 125000  # Mbps to bytes/s
        
        A_n = alpha + self.L0_bytes / b_dn if b_dn > 0 else alpha
        B_n = t_lora / self.r_max + self.unit_lora_bytes / b_up if b_up > 0 else t_lora / self.r_max
        
        return A_n, B_n
    
    def _solve_p1_cvxpy(
        self,
        round_idx: int,
        r_i: float,
        client_ids: List[int],
        N: int,
        target_sum: Optional[int] = None,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Solve P1 using cvxpy convex optimization with per-client cap constraints.
        """
        import cvxpy as cp

        r = cp.Variable(N)
        T = cp.Variable()

        A = []
        B = []
        upper_bounds = []
        for client_id in client_ids:
            A_n, B_n = self._get_time_coefficients(client_id)
            A.append(A_n)
            B.append(B_n)
            upper_bounds.append(float(self._get_client_rank_cap(client_id)))

        if target_sum is None:
            target_sum = int(round(N * r_i))

        constraints = [
            r >= self.r_min,
            r <= np.asarray(upper_bounds, dtype=float),
            cp.sum(r) == float(target_sum),
        ]

        for n in range(N):
            constraints.append(T >= A[n] + B[n] * r[n])

        prob = cp.Problem(cp.Minimize(T), constraints)

        try:
            prob.solve(solver=cp.Clarabel)
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception as e:
                logger.warning(f"cvxpy solve failed: {e}")
                return self._solve_p1_heuristic(round_idx, r_i, client_ids, N, target_sum)

        if r.value is None:
            logger.warning("cvxpy returned no solution, using heuristic")
            return self._solve_p1_heuristic(round_idx, r_i, client_ids, N, target_sum)

        r_continuous = r.value
        r_int = self._round_to_valid_integers(
            r_continuous=r_continuous,
            target_sum=target_sum,
            upper_bounds=upper_bounds,
        )

        r_per_client = {}
        r_hat_per_client = {}
        for idx, client_id in enumerate(client_ids):
            client_cap = self._get_client_rank_cap(client_id)
            r_per_client[client_id] = r_int[idx]
            r_hat_per_client[client_id] = max(self.r_min, min(client_cap, r_int[idx] - 1))

        self.current_ranks = r_per_client.copy()
        self.current_hat_ranks = r_hat_per_client.copy()

        logger.info(
            f"Round {round_idx} Stage 2 (cvxpy, cap-aware): "
            f"ranks={list(r_per_client.values())}, avg={np.mean(list(r_per_client.values())):.2f}"
        )

        return r_per_client, r_hat_per_client

    def _solve_p1_heuristic(
        self,
        round_idx: int,
        r_i: float,
        client_ids: List[int],
        N: int,
        target_sum: Optional[int] = None,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Solve P1 using a heuristic approach with immutable per-client caps.

        Strategy: assign higher ranks to faster clients (lower B_n coefficient)
        while respecting each client's local scalar rank cap.
        """
        latency_coeffs = []
        client_caps = {}
        for client_id in client_ids:
            _, B_n = self._get_time_coefficients(client_id)
            latency_coeffs.append((client_id, B_n))
            client_caps[client_id] = self._get_client_rank_cap(client_id)

        latency_coeffs.sort(key=lambda x: x[1])

        if target_sum is None:
            target_sum = int(round(N * r_i))

        seed_rank = int(round(r_i))
        r_assigned = {
            client_id: max(self.r_min, min(client_caps[client_id], seed_rank))
            for client_id in client_ids
        }
        current_sum = sum(r_assigned.values())

        diff = target_sum - current_sum
        idx = 0
        max_iterations = max(1, sum(client_caps[cid] - self.r_min for cid in client_ids)) * max(1, N)

        if diff > 0:
            while diff > 0 and idx < max_iterations:
                client_id = latency_coeffs[idx % N][0]
                if r_assigned[client_id] < client_caps[client_id]:
                    r_assigned[client_id] += 1
                    diff -= 1
                idx += 1
        elif diff < 0:
            while diff < 0 and idx < max_iterations:
                client_id = latency_coeffs[-(idx % N) - 1][0]
                if r_assigned[client_id] > self.r_min:
                    r_assigned[client_id] -= 1
                    diff += 1
                idx += 1

        for client_id in r_assigned:
            r_assigned[client_id] = max(self.r_min, min(client_caps[client_id], r_assigned[client_id]))

        r_per_client = r_assigned
        r_hat_per_client = {
            cid: max(self.r_min, min(client_caps[cid], rank - 1))
            for cid, rank in r_assigned.items()
        }

        self.current_ranks = r_per_client.copy()
        self.current_hat_ranks = r_hat_per_client.copy()

        logger.info(
            f"Round {round_idx} Stage 2 (heuristic, cap-aware): "
            f"ranks={list(r_per_client.values())}, avg={np.mean(list(r_per_client.values())):.2f}"
        )

        return r_per_client, r_hat_per_client

    def _round_to_valid_integers(
        self,
        r_continuous: np.ndarray,
        target_sum: int,
        upper_bounds: List[int],
    ) -> List[int]:
        """
        Round continuous ranks to integers while preserving the feasible target sum
        under per-client upper bounds.
        """
        upper_bounds = np.asarray(upper_bounds, dtype=int)

        r_int = np.round(r_continuous).astype(int)
        r_int = np.clip(r_int, self.r_min, upper_bounds)

        current_sum = int(np.sum(r_int))
        diff = int(target_sum - current_sum)
        residuals = r_continuous - r_int

        if diff > 0:
            indices = np.argsort(-residuals)
            for idx in indices:
                if diff <= 0:
                    break
                if r_int[idx] < upper_bounds[idx]:
                    r_int[idx] += 1
                    diff -= 1
        elif diff < 0:
            indices = np.argsort(residuals)
            for idx in indices:
                if diff >= 0:
                    break
                if r_int[idx] > self.r_min:
                    r_int[idx] -= 1
                    diff += 1

        return r_int.tolist()

    def get_current_average_rank(self) -> float:
        """Get the current average rank r_i."""
        return self.r_history[-1] if self.r_history else float(self.init_rank)
    
    def is_in_warmup(self, round_idx: int) -> bool:
        """Check if we're still in warm-up phase."""
        return round_idx < self.warmup_rounds
    
