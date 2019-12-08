import records

class DBEngine:
    def __init__(self, db_file):
        self.db = records.Database('sqlite:///{}'.format(db_file))
        self.conn = self.db.get_connection()

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith("table"):
            table_id = "table_{}".format(table_id.replace('-','_'))
        table_info = self.conn.query("SELECT sql from sqlite_master WHERE tbl_name = :name", name=table_id)


ex_gold = engine.execute_query(gold_example['table_id'], lf_gold_query, lower=True)
table_id = gold_example['table_id']
table_info = engine.conn.query("SELECT sql from sqlite_master WHERE tbl_name = :name", name=table_id).all()[0].sql
select_index = lf_gold_query.sel_index

