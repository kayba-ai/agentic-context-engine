from ace.core.recursive_agent import AgenticConfig
from ace.core.skillbook import Skillbook
from ace.implementations.sm_tools import SMDeps


def test_recursive_child_state_is_not_shared_with_parent():
    parent = SMDeps(config=AgenticConfig(), skillbook=Skillbook())
    seed = parent.skillbook.add_skill(
        section="strategy", issue="seed", keywords=["seed"], insight="seed"
    )
    child = SMDeps(
        config=parent.config,
        skillbook=parent.skillbook,
        operations=parent.operations,
    )

    # Mirror the state-isolation contract used by register_recurse.
    child.skillbook = child.skillbook.__class__.loads(child.skillbook.dumps())
    child.operations = []
    child_skill = child.skillbook.add_skill(
        section="strategy", issue="child-only", keywords=["child"], insight="child"
    )
    child.skillbook.update_skill(
        seed.id, issue="child seed", keywords=["seed"], insight="child seed"
    )
    child.operations.append("child-only")

    assert parent.skillbook.get_skill(seed.id).issue == "seed"
    assert parent.skillbook.get_skill(child_skill.id) is None
    assert parent.operations == []
