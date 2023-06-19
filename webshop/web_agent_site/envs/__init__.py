from gym.envs.registration import register

from webshop.web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from webshop.web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

register(
    id="WebAgentSiteEnv-v0",
    entry_point="web_agent_site.envs:WebAgentSiteEnv",
)

register(
    id="WebAgentTextEnv-v0",
    entry_point="web_agent_site.envs:WebAgentTextEnv",
)
