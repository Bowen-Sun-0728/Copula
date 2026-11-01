"""The one typed error that crosses process boundaries: a kind (the rejection) and a detail."""


class EvalrtError(Exception):
    def __init__(self, kind, detail=''):
        super().__init__('%s: %s' % (kind, detail))
        self.kind = kind
        self.detail = detail
