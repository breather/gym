"""
Set indices for a database table to better handle the queries performed against it
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

    def __init__(self, database, host, user, password, port, table, schema):
        # Database connection setup
        self.database = database
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.closed = True
        self.conn = self.connect_db()
        # Target that we are optimizing for
        self.schema = schema
        self.table = table
        self.column_stats = None

        # Execution times to calculate reward
        self.initial_execution_time = None
        self.previous_execution_time = None

        # Optimize table on these queries
        self.queries = []
        self.query_plans = []  # TODO: optimize based on query plans

        self.action_space = None
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
        Action space consists of every column in table
        with each column having a boolean for whether to include it in the index
        and a rank for sorting the order of the column (if included).

        TODO: need to figure out how to handle when multiple columns are sampled
        to have the same rank
        '''
        if self.closed: self.connect_db()

        self.column_stats = pd.read_sql('''
            SELECT *
              FROM pg_stats
             WHERE schemaname = '{schema}'
               AND tablename = '{table}'
        '''.format(schema=self.schema, table=self.table), self.conn)

        assert self.column_stats.attname.is_unique, 'columns are not unique'

        self.action_space = spaces.Dict({
            colname: spaces.Dict({
                'is_included': spaces.Discrete(2),
                'rank': spaces.Box(low=0, high=1, shape=(1,))
            }) for colname in self.column_stats.attname
        })

    def create_index(self, action):
        '''
        Convert action to SQL

        Each column has a boolean flag indicating whether it should be
        used in the index and a rank variable that is used for
        determining their order in the index
        '''
        included_cols = []
        for col, col_action in sorted(action.iteritems(),
                                      key=lambda (c,a): a['rank'],
                                      reverse=True):
            if col_action['is_included'] == 1:
                included_cols.append('"{}"'.format(col))
        stmt = 'CREATE INDEX ON {schema}.{table} ({cols})'.format(cols=', '.join(included_cols),
                                                                  schema=self.schema,
                                                                  table=self.table)
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
        '''
        ncol = self.column_stats.attname.shape[0]
        nquery = len(self.queries)
        obs_shape = (nquery, 1 + (ncol * 2))
        return spaces.Box(
            low=np.zeros(obs_shape),
            high=np.array([np.inf] * obs_shape[0] * obs_shape[1]).reshape(obs_shape)
        )

    def get_cardinality(self):
        '''
        Returns a pandas dataframe with columns attname (aka colname) and n_distinct
        '''
        colstat = self.column_stats[['attname', 'n_distinct']]
        if colstat.n_distinct.min() < 0:
            # n_distinct is negative if Postgres believes the cardinality is proportional to row_count
            # Replace negative n_distinct values with approximation of actual cardinality
            plan = self.explain("SELECT * FROM {schema}.{table}".format(schema=self.schema, table=self.table))
            nrow = plan['Plan']['Plan Rows']
            replacement = colstat.n_distinct * nrow * -1
            colstat['n_distinct'] = replacement.where(colstat.n_distinct < 0, other=colstat.n_distinct)
            assert colstat.n_distinct.min() >= 0
        return colstat

    def get_column_cost(self, query):
        '''
        Return column presence in the query plan. If found, attribute the
        Total Cost - Startup Cost to this column.

        '''
        columns = pd.DataFrame(self.column_stats.attname)
        columns['cost'] = 0
        for col in columns.attname:
            plan = self.explain(query)

    def preprocess_query_plan(plan):
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
        return flat_df

    def explain(self, query):
        '''
        Return the results of a SQL EXPLAIN query as a dict
        '''
        cur = self.conn.cursor()
        stmt = 'EXPLAIN (FORMAT JSON) ' + query
        cur.execute(stmt)
        return cur.fetchone()[0][0]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        # Undo all changes made by model
        self.conn.rollback()
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        pass
