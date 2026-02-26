from ace.llm_providers import LiteLLMClient
from ace.skillbook import Skillbook
from ace2.pipelines import OfflineACE, OnlineACE

client = LiteLLMClient(model="gpt-4o-mini")
skillbook = Skillbook()

# --- Offline (train over samples with epochs) ---
ace = OfflineACE.from_client(client, skillbook=skillbook)
results = ace.run(train_samples, environment, epochs=3)

# --- Online (stream new samples) ---
ace = OnlineACE.from_client(client, skillbook=skillbook)
results = ace.run(test_samples, environment)

# --- Custom roles when you need control ---
from ace.roles import Agent, Reflector, SkillManager
from ace.prompts_v2_1 import PromptManager

pm = PromptManager()
ace = OfflineACE.from_roles(
    agent=Agent(client, prompt_template=pm.get_agent_prompt()),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    skillbook=skillbook,
    reflection_window=5,
)
results = ace.run(samples, environment, epochs=2)
