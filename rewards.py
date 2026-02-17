import math

class RewardCalculator:
    """
    Implements the core AURORA reward dynamics.
    Supports configurable penalties for ablation studies.
    """
    def __init__(self, use_traceability=True, attack_weight=5.0):
        self.use_traceability = use_traceability
        self.attack_weight = attack_weight

    def calculate_vlm_reward(self, correct_claims, incorrect_claims, total_claims, description_length):
        """
        Reward for the Generator (VLM).
        """
        if total_claims == 0:
            return 0.0

        accuracy_score = (correct_claims - incorrect_claims) / total_claims
        bonus_score = 0.5 * math.log(correct_claims + 1)
        
        length_penalty = 0.0
        if description_length < 20:
            length_penalty = -2.0
            
        total = accuracy_score + bonus_score + length_penalty
        return total

    def calculate_verifier_reward(self, claim_status, claim_traceable, correlation_score=0.0):
        """
        Reward for the Verifier (Opponent).
        """
        reward = 0.0
        
        # 1. Traceability Check (Ablation: can be disabled)
        if self.use_traceability and not claim_traceable:
            return -10.0
            
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
        reward -= (correlation_score * 2.0)
            
        return reward
