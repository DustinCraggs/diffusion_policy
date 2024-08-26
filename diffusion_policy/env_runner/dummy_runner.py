from typing import Dict

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class DummyRunner(BaseImageRunner):

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
