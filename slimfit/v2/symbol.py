import sympy as sp


class Symbols(dict):
    def __init__(self, names, cls=sp.Symbol, **kwargs):
        default_kwargs = {"seq": True}
        default_kwargs.update(kwargs)
        super().__init__({s.name: s for s in sp.symbols(names, cls=cls, **default_kwargs)})

    def __repr__(self):
        return f"Symbols({list(self.keys())})"

    def __getattr__(self, name) -> sp.Symbol:
        if name in self:
            return self[name]
        raise AttributeError(f"'SymbolNamespace' object has no attribute '{name}'")
