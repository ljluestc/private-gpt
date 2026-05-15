from pydantic import BaseModel

import private_gpt  # noqa: F401


def test_update_forward_refs_accepts_localns_for_compatibility() -> None:
    class Child(BaseModel):
        value: str

    class Parent(BaseModel):
        child: "Child | None" = None

    Parent.update_forward_refs(Child=Child)

    validated = Parent.model_validate({"child": {"value": "ok"}})
    assert validated.child is not None
    assert validated.child.value == "ok"
