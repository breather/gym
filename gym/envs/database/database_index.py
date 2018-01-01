"""
Set indices for a database table to better handle the queries performed against it
TODO: modify to work on a single query at a time (for transfer learning)
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import psycopg2

from keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)

class DatabaseIndexEnv(gym.Env):

    def __init__(self):
        # Database connection setup
        self.conn = None
        # Optimize table on these queries
        # DataFrame w/ following columns: query, schema, table, frequency, priority
        # frequency and priority might end up as features in the state
        self.queries = None
        self.current_index = 0
        # contains various statistics stored by the database on its tables
        self.column_stats = None
        self.table_rows = None

        # Execution times to calculate reward
        self.previous_cost = None

        self.action_space = spaces.Tuple((
            spaces.MultiBinary(32),
            spaces.Box(low=0, high=1, shape=(32,))
        ))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2*32+1,))

        self._seed()
        self.state = None

        self.steps_beyond_done = None

    def connect_db(self, database, host, user, password, port):
        # TODO: replace with raise NotImplementedError
        self.conn = psycopg2.connect(
            database=database,
            user=user,
            password=password,
            host=host,
            port=port,
        )

    def get_stats(self):
        tables = self.queries['table'].unique()
        schemas = self.queries['schema'].unique()
        self.column_stats = pd.read_sql('''
            SELECT pg_stats.schemaname
                 , pg_stats.tablename
                 , pg_stats.attname AS colname
                 , CASE
                   WHEN pg_stats.n_distinct >= 0 THEN pg_stats.n_distinct
                   -- otherwise n_distinct is the negative proportion of cardinality
                   ELSE -1 * pg_stats.n_distinct * pg_stat_user_tables.n_live_tup
                    END AS n_distinct
                 , pg_stat_user_tables.n_live_tup AS n_row
              FROM pg_stats
              JOIN pg_stat_user_tables
                ON pg_stats.schemaname = pg_stat_user_tables.schemaname
               AND pg_stats.tablename = pg_stat_user_tables.relname
             WHERE pg_stats.schemaname IN ({schemas})
               AND pg_stats.tablename IN ({tables})
        '''.format(schemas=','.join("'{0}'".format(s) for s in schemas),
                   tables=','.join("'{0}'".format(t) for t in tables)),
            self.conn)
        self.table_rows = self.column_stats[['schemaname', 'tablename', 'n_row']].drop_duplicates()

    def create_index(self, action, schema, table):
        '''
        Convert action to SQL

        Each column has a boolean flag indicating whether it should be
        used in the index and a rank variable that is used for
        determining their order in the index
        '''
        padded_action = pad_sequences(action,
                                      maxlen=self.column_cost.shape[0],
                                      padding='post',
                                      truncating='post',
                                      dtype='float64')
        self.column_cost['is_included'] = padded_action[0]
        self.column_cost['rank'] = padded_action[1]

        included_cols = self.column_cost \
            .sort('rank', ascending=False) \
            .loc[self.column_cost['is_included'] == 1, 'colname']

        if len(included_cols) > 0:
            stmt = 'CREATE INDEX ON {schema}.{table} ({cols})'.format(cols=', '.join(included_cols),
                                                                      schema=schema,
                                                                      table=table)
            print 'Running.... ' + stmt
            self.conn.cursor().execute(stmt)
            return stmt

    def query_indexes(self, query_record):
        '''
        Query indexes on the target table

        Useful for seeing cumulative actions and
        for checking whether rollback worked
        '''
        query, schema, table, frequency = self.parse_query_record(query_record)
        return pd.read_sql('''
            SELECT *
              FROM pg_indexes
             WHERE schemaname = '{schema}'
               AND tablename = '{table}'
        '''.format(schema=schema, table=table), self.conn)

    def get_cardinality(self):
        '''
        Returns a pandas dataframe with columns colname (aka colname) and n_distinct
        '''
        colstat = self.column_stats[['colname', 'n_distinct']]
        if colstat.n_distinct.min() < 0:
            # n_distinct is negative if Postgres believes the cardinality is proportional to row_count
            # Replace negative n_distinct values with approximation of actual cardinality
            plan = self.explain("SELECT * FROM {schema}.{table}".format(schema=self.schema, table=self.table))
            nrow = plan['Plan']['Plan Rows']
            replacement = colstat.n_distinct * nrow * -1
            colstat['n_distinct'] = replacement.where(colstat.n_distinct < 0, other=colstat.n_distinct)
            assert colstat.n_distinct.min() >= 0
        return colstat

    def get_column_cost(self, schema, table, query):
        '''
        Return column presence in the query plan. If found, attribute the
        Total Cost - Startup Cost to this column.
        '''
        plan = self.explain(query)
        flat = self.preprocess_query_plan(plan)
        columns = self.column_stats.loc[(self.column_stats['schemaname'] == schema).values &
                                            (self.column_stats['tablename'] == table).values,
                                        ['colname', 'n_distinct']]
        columns['cost'] = 0
        for col in columns.colname:
            isin_filter = flat['Filter'].str.contains(col).fillna(False)
            isin_index = flat['Index Cond'].str.contains(col).fillna(False)
            indicator = isin_filter | isin_index
            if indicator.any():
                columns.loc[columns['colname'] == col, 'cost'] = flat[indicator]['Node Cost'].sum()
        self.column_cost = columns.sort('n_distinct', ascending=False)

    def preprocess_query_plan(self, plan):
        '''
        The query plan is a nested dictionary which is hard to work with.
        Flatten the nest into a table of nodes with relevant information
        '''
        flat = [plan['Plan']]
        def flatten_plan(nodes):
            '''
            Move all the nested nodes so they are at the same level
            '''
            for node in nodes:
                if 'Plans' in node:
                    plans = node.pop('Plans')
                    flat.append(plans[0])
                    flatten_plan(plans)
        flatten_plan(flat)
        flat_df = pd.DataFrame(flat)
        # The incremental cost introduced by a given node
        flat_df['Node Cost'] = flat_df['Total Cost'] - flat_df['Startup Cost']
        # Remove nodes that did not involve filtering on a column
        # TODO: consider including 'Sort Key'
        for col in ['Filter', 'Index Cond']:
            if not col in flat_df.columns:
                flat_df[col] = None
        return flat_df[~(
            flat_df['Filter'].isnull() &
            flat_df['Index Cond'].isnull()
        )]

    def explain(self, query):
        '''
        Return the results of a SQL EXPLAIN query as a dict
        '''
        cur = self.conn.cursor()
        stmt = 'EXPLAIN (FORMAT JSON) ' + query
        cur.execute(stmt)
        return cur.fetchone()[0][0]

    def set_queries(self, queries):
        '''
        queries is a DataFrame containing queries and their frequency
        '''
        assert 'query' in queries.columns
        assert 'frequency' in queries.columns
        self.queries = queries
        self.get_stats()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        print 'step start'
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if self.steps_beyond_done is None:
            current_query_record = self.queries.loc[self.current_index]
            query, schema, table, frequency = self.parse_query_record(current_query_record)

            previous_cost = self.previous_cost  # cost before action
            index_stmt = self.create_index(action, schema, table)  # action
            cost_after = self.get_query_cost(query)  # cost after action

            complexity_penalty = 0  # unnecessary if you run test over insert queries as well
            reward = frequency * (previous_cost - cost_after - complexity_penalty)

            self.current_index += 1
            done = self.current_index == self.queries.shape[0]
            if not done:
                next_query_record = self.queries.loc[self.current_index]
                self.set_state(next_query_record)
            else:
                self.state = np.zeros_like(self.state)
                self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.state = np.zeros_like(self.state)
            reward = 0.0
            done = True

        return self.state, reward, done, {'indexdef': index_stmt}

    def _reset(self):
        print 'reset'
        # Undo all changes made by model
        self.conn.rollback()
        self.random_sort_queries()
        self.current_index = 0
        first_query_record = self.queries.loc[self.current_index]
        self.set_state(first_query_record)
        self.steps_beyond_done = None
        return self.state

    def set_state(self, query_record):
        '''
        The features that are observed, which are affected by actions
        '''
        query, schema, table, frequency = self.parse_query_record(query_record)

        self.get_column_cost(schema, table, query)
        self.previous_cost = self.get_query_cost(query)

        values = []
        for col in ['n_distinct', 'cost']:
            values.append(self.column_cost[col].tolist())
        # No matter the table, state will always be the same shape
        padded_vals = pad_sequences(values,
            maxlen=32,
            padding='post',
            truncating='post',
            dtype='float64')
        # Include additional features, such as query frequency
        state = np.append(padded_vals.flatten(), frequency)
        self.state = state

    def parse_query_record(self, query_record):
        query = query_record['query']
        schema = query_record['schema']
        table = query_record['table']
        frequency = query_record['frequency']
        return (query, schema, table, frequency)

    def _render(self, mode='human', close=False):
        pass

    def get_query_cost(self, query):
        return self.explain(query)['Plan']['Total Cost']

    def random_sort_queries(self):
        '''
        Re-order queries so the model does not fit on it
        '''
        self.queries = self.queries.sample(frac=1).reset_index(drop=True)
