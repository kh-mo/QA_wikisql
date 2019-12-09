class Query:
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE',
            'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']
    def __init__(self, sel_index, agg_index, conditions=tuple(), ordered=False):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.conditions = list(conditions)
        self.ordered = ordered

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(agg=self.agg_ops[self.agg_index],
                                                     sel='col{}'.format(self.sel_index),)
        if self.conditions:
            rep += ' WHERE ' + ' AND '.join(
                ['{} {} {}'.format('col{}'.format(idx), self.cond_ops[oper], value) for idx, oper, value in self.conditions])
        return rep

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            indices = self.sel_index == other.sel_index and self.agg_index == other.agg_index
            if other.ordered:
                conds = [(col, op, str(cond).lower()) for col, op, cond in self.conditions] == [
                    (col, op, str(cond).lower()) for col, op, cond in other.conditions]
            else:
                conds = set([(col, op, str(cond).lower()) for col, op, cond in self.conditions]) == set(
                    [(col, op, str(cond).lower()) for col, op, cond in other.conditions])

            return indices and conds
        return NotImplemented

    @classmethod
    def from_dict(cls, example, ordered=False):
        return cls(sel_index=example['sel'], agg_index=example['agg'], conditions=example['conds'], ordered=ordered)
