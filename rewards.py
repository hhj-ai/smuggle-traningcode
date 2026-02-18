import math

class RewardCalculator:
    """
    Implements the core AURORA reward dynamics.
    Supports configurable penalties for ablation studies.
    """
    def __init__(self, use_traceability=True, attack_weight=5.0,
                 bonus_beta=0.5, correlation_weight=2.0,
                 length_threshold=20, length_penalty=-2.0):
        self.use_traceability = use_traceability
        self.attack_weight = attack_weight
        self.bonus_beta = bonus_beta
        self.correlation_weight = correlation_weight
        self.length_threshold = length_threshold
        self.length_penalty = length_penalty

    def calculate_vlm_reward(self, correct_claims, incorrect_claims, total_claims, description_length):
        """
        Reward for the Generator (VLM).
        """
        if total_claims == 0:
            return 0.0

        accuracy_score = (correct_claims - incorrect_claims) / total_claims
        bonus_score = self.bonus_beta * math.log(correct_claims + 1)

        lp = 0.0
        if description_length < self.length_threshold:
            lp = self.length_penalty

        total = accuracy_score + bonus_score + lp
        return total

    def calculate_verifier_reward(self, claim_status, claim_traceable, correlation_score=0.0):
        """
        Reward for the Verifier (Opponent).
        """
        correlation_penalty = correlation_score * self.correlation_weight

        # 1. Traceability Check (Ablation: can be disabled)
        if self.use_traceability and not claim_traceable:
            return -10.0 - correlation_penalty

        reward = 0.0

        # 2. Attack Success (Configurable weight)
        if claim_status == 'incorrect':
            reward = self.attack_weight

        # 3. Verification Success
        elif claim_status == 'correct':
            reward = 1.0

        # 4. Uncertainty
        elif claim_status == 'uncertain':
            reward = -0.1

        # 5. Correlation Penalty
        reward -= correlation_penalty

        return reward
