"""
Linear agent with state count reward bonuses
"""

from agents.state_action_bonus_agent import (QAgent as QBonusAgent,
                                             SARSAAgent as SBonusAgent)


def state_bonus(Parent):
    class Mixin(Parent):
        def compute_bonus(self, s, a):
            return super().compute_bonus(s, 0)
    return Mixin


QAgent = state_bonus(QBonusAgent)
SARSAAgent = state_bonus(SBonusAgent)
