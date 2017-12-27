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

logger = logging.getLogger(__name__)

class DatabaseIndexEnv(gym.Env):

    def __init__(self, database, host, user, password, port, table, schema, queries):
        # Database connection setup
        self.database = database
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.closed = True
        self.conn = self.connect_db()
        # Optimize table on these queries
        # DataFrame w/ following columns: query, schema, table, frequency, priority
        # frequency and priority might end up as features in the state
        self.queries = queries
        self.current_index = 0
        # contains various statistics stored by the database on its tables
        self.column_stats = None
        self.table_rows = None
        self.get_stats()

        # Execution times to calculate reward
        self.previous_cost = None

        self.action_space = self.determine_action_space()
        self.observation_space = self.determine_observation_space()

        # Remember previous actions taken for rollback and also more efficient searches
        self.actions_taken = []

        self._seed()
        self.state = None

        self.steps_beyond_done = None

    def connect_db(self):
        # TODO: replace with raise NotImplementedError
        self.conn = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )
        self.closed = self.conn.closed

    def determine_action_space(self):
        '''
        Action space consists of up to 32 columns with a binary flag to indicate
        if the column should be included in the index or not and a continuous
        rank to indicate what order the columns should be in the index.

        If a table has more than 32 columns, select the ones with highest cardinality
        for the action space.

        TODO: need to figure out how to handle when multiple columns are sampled
        to have the same rank
        '''
        self.action_space = spaces.Tuple((
            spaces.MultiBinary(32),
            spaces.Box(low=0, high=1, shape=(32,))
        ))

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
        '''.format(schemas=','.join("'{0}}'".format(s) for s in schemas),
                   tables=','.join("'{0}'".format(t) for t in table)),
            self.conn)
        self.table_rows = self.column_stats[['schemaname', 'tablename', 'n_row']].drop_duplicates()

    def create_index(self, action, schema, table):
        '''
        Convert action to SQL

        Each column has a boolean flag indicating whether it should be
        used in the index and a rank variable that is used for
        determining their order in the index
        '''
        self.column_cost['is_included'] = action[0]
        self.column_cost['rank'] = action[1]

        included_cols = self.column_cost \
            .sort('rank', ascending=False) \
            .loc[self.column_cost['is_included'] == 1, 'colname']

        if len(included_cols) > 0:
            stmt = 'CREATE INDEX ON {schema}.{table} ({cols})'.format(cols=', '.join(included_cols),
                                                                      schema=schema,
                                                                      table=table)
            self.conn.cursor().execute(stmt)

    def query_indexes(self):
        '''
        Query indexes on the target table

        Useful for seeing cumulative actions and
        for checking whether rollback worked
        '''
        return pd.read_sql('''
            SELECT *
              FROM pg_indexes
             WHERE schemaname = '{schema}'
               AND tablename = '{table}'
        '''.format(schema=self.schema, table=self.table))

    def determine_observation_space(self):
        '''
        Observation space consists of one row per query with the following information:
        - for each column, what is its cardinality? (unchanging wrt to query and action)
        - for each column, what was the cost associated with it in the query?
        - how many steps were involved in the query?

        # TODO: modify to work on a single query (Probably a Tuple this time)
        '''
        ncol = self.column_stats.colname.shape[0]
        nquery = len(self.queries)
        obs_shape = (nquery, 1 + (ncol * 2))
        return spaces.Box(
            low=np.zeros(obs_shape),
            high=np.array([np.inf] * obs_shape[0] * obs_shape[1]).reshape(obs_shape)
        )

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
        columns = self.column_stats.loc[self.column_stats['schemaname'] == schema &
                                            self.column_stats['tablename'] == table,
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
                    flat.append(node)
                    flatten_plan(plans)
                else:
                    flat.append(node)
        flatten_plan(flat)
        flat_df = pd.DataFrame(flat)
        # The incremental cost introduced by a given node
        flat_df['Node Cost'] = flat_df['Total Cost'] - flat_df['Startup Cost']
        # Remove nodes that did not involve filtering on a column
        flat_df[~(flat_df['Filter'].isnull() & flat_df['Index Cond'].isnull())]
        return flat_df

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

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        print 'step start'
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        current_query_record = self.queries.loc[self.current_index]
        query, schema, table, frequency = self.parse_query_record(current_query_record)

        previous_cost = self.previous_cost  # cost before action
        self.create_index(action, schema, table)  # action
        cost_after = self.get_query_cost(query)  # cost after action

        complexity_penalty = 0  # unnecessary if you run test over insert queries as well
        reward = previous_cost - cost_after - complexity_penalty

        self.current_index += 1
        next_query_record = self.queries.loc[self.current_index]
        self.set_state(next_query_record)

        done = self.current_index == self.queries.shape[0]

        return np.array(self.state), reward, done, {}

    def _reset(self):
        print 'reset'
        # Undo all changes made by model
        self.conn.rollback()
        self.random_sort_queries()
        self.current_index = 0
        first_query_record = self.queries.loc[self.current_index]
        self.set_state(first_query_record)
        self.steps_beyond_done = None
        return np.array(self.state)

    def set_state(self, query_record):
        '''
        The features that are observed, which are affected by actions
        '''
        query, schema, table, frequency = self.parse_query_record(query_record)

        self.get_column_cost(schema, table, query)
        self.previous_cost = self.get_query_cost(query)

        self.state = column_cost \
            [['n_distinct', 'cost']] \
            .head(32)

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
