#!/usr/bin/env python3
"""Offline fixture trace through the real environment; not a policy/training run."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ReasoningEconomicsEnv'))
from data.question import Question
from env.config import EnvConfig
from env.episode_sampler import EpisodeSampler
from env.models import ReasonBudgetAction
from env.reason_budget_env import ReasonBudgetEnvironment

QUESTIONS = [
    Question('demo-1', 'Compute 12000 + 345.', '12345', 'fixture', 'hand-authored demo'),
    Question('demo-2', 'Compute 45000 + 678.', '45678', 'fixture', 'hand-authored demo'),
]

class WhitespaceTokenizer:
    """Explicit fixture only: counts whitespace pieces, not model tokens."""
    def encode(self, text, add_special_tokens=False):
        return text.split()

def run():
    trace = {
        'scope': 'CPU fixture smoke, no model generation, network server or training',
        'substitutions': ['fixed two-question sampler', 'whitespace tokenizer', 'scripted responses', 'injected verifier exception in one case'],
        'observations': [], 'cases': {},
    }
    with patch.object(EpisodeSampler, 'sample_episode', return_value=QUESTIONS), patch.object(ReasonBudgetEnvironment, '_get_tokenizer', return_value=WhitespaceTokenizer()):
        def reset():
            env = ReasonBudgetEnvironment(EnvConfig(num_questions=2, total_budget=40, min_tokens=1))
            obs = env.reset(total_budget=40, episode_id='fixture-demo')
            return env, obs

        for name, responses in [
            ('correct', [r'\boxed{12345}', r'\boxed{45678}']),
            ('wrong', [r'\boxed{0}', r'\boxed{0}']),
            ('malformed', ['', '']),
        ]:
            env, obs = reset()
            visible = obs.model_dump()
            trace['observations'].append(visible)
            assert all(q.answer not in json.dumps(visible) for q in QUESTIONS)
            steps = []
            for response in responses:
                obs = env.step(ReasonBudgetAction(response=response))
                steps.append({'response': response, 'observation': obs.model_dump()})
            trace['cases'][name] = {'steps': steps, 'done': obs.done, 'total_correct': env.state.total_correct, 'state': env.state.model_dump()}

        env, obs = reset()
        with patch('env.reason_budget_env.grade_answer', side_effect=RuntimeError('injected demo verifier failure')):
            try:
                env.step(ReasonBudgetAction(response=r'\boxed{12345}'))
            except RuntimeError as exc:
                # Label belongs to this example harness. Environment propagates
                # the exception; it does not return a structured error reward.
                trace['cases']['verifier_error'] = {
                    'classification_source': 'example harness, not environment schema',
                    'error_type': type(exc).__name__, 'error': str(exc),
                    'reward': None, 'recorded_as_scored_answer': env.state.questions_answered != 0,
                }
        trace['fixture_gold_not_in_reset_observation'] = True
    return trace

if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
