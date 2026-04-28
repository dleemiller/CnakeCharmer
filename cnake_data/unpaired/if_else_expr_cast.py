class Foo:
    def __repr__(self):
        return "<Foo>"


def test_type_cast(obj, cond):
    return [obj] if cond else obj
