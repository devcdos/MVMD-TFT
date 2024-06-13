# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes used for hyperparameter optimisation.

Two main classes exist:
1) HyperparamOptManager used for optimisation on a single machine/GPU.
2) DistributedHyperparamOptManager for multiple GPUs on different machines.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import libs.utils as utils
import numpy as np
import pandas as pd


import random
import math
Deque = collections.deque


class HyperparamOptManager:
  """Manages hyperparameter optimisation using random search for a single GPU.

  Attributes:
    param_ranges: Discrete hyperparameter range for random search.
    results: Dataframe of validation results.
    fixed_params: Fixed model parameters per experiment.
    saved_params: Dataframe of parameters trained.
    best_score: Minimum validation loss observed thus far.
    optimal_name: Key to best configuration.
    hyperparam_folder: Where to save optimisation outputs.
  """

  def __init__(self,
               param_ranges,
               fixed_params,
               model_folder,
               override_w_fixed_params=True):
    """Instantiates model.

    Args:
      param_ranges: Discrete hyperparameter range for random search.
      fixed_params: Fixed model parameters per experiment.
      model_folder: Folder to store optimisation artifacts.
      override_w_fixed_params: Whether to override serialsed fixed model
        parameters with new supplied values.
    """

    self.param_ranges = param_ranges

    self._max_tries = 1000
    self.results = pd.DataFrame()
    self.fixed_params = fixed_params
    self.saved_params = pd.DataFrame()

    self.best_score = np.Inf
    self.optimal_name = ""

    # Setup
    # Create folder for saving if its not there
    self.hyperparam_folder = model_folder
    utils.create_folder_if_not_exist(self.hyperparam_folder)

    self._override_w_fixed_params = override_w_fixed_params

  def load_results(self):
    """Loads results from previous hyperparameter optimisation.

    Returns:
      A boolean indicating if previous results can be loaded.
    """
    print("Loading results from", self.hyperparam_folder)

    results_file = os.path.join(self.hyperparam_folder, "results.csv")
    params_file = os.path.join(self.hyperparam_folder, "params.csv")

    if os.path.exists(results_file) and os.path.exists(params_file):

      self.results = pd.read_csv(results_file, index_col=0)
      self.saved_params = pd.read_csv(params_file, index_col=0)

      if not self.results.empty:
        self.results.at["loss"] = self.results.loc["loss"].apply(float)
        self.best_score = self.results.loc["loss"].min()

        is_optimal = self.results.loc["loss"] == self.best_score
        self.optimal_name = self.results.T[is_optimal].index[0]

        return True

    return False

  def _get_params_from_name(self, name):
    """Returns previously saved parameters given a key."""
    params = self.saved_params

    selected_params = dict(params[name])
    if self._override_w_fixed_params:
      for k in self.fixed_params:
        selected_params[k] = self.fixed_params[k]

    return selected_params

  def get_best_params(self):
    """Returns the optimal hyperparameters thus far."""

    optimal_name = self.optimal_name

    return self._get_params_from_name(optimal_name)

  def clear(self):
    """Clears all previous results and saved parameters."""
    shutil.rmtree(self.hyperparam_folder)
    os.makedirs(self.hyperparam_folder)
    self.results = pd.DataFrame()
    self.saved_params = pd.DataFrame()

  def _check_params(self, params):
    """Checks that parameter map is properly defined."""

    valid_fields = list(self.param_ranges.keys()) + list(
        self.fixed_params.keys())
    invalid_fields = [k for k in params if k not in valid_fields]
    missing_fields = [k for k in valid_fields if k not in params]

    if invalid_fields:
      raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(
          invalid_fields, valid_fields))
    if missing_fields:
      raise ValueError("Missing Fields Found {} - Valid ones are {}".format(
          missing_fields, valid_fields))

  def _get_name(self, params):
    """Returns a unique key for the supplied set of params."""

    self._check_params(params)

    fields = list(params.keys())
    fields.sort()

    return "_".join([str(params[k]) for k in fields])

  def get_next_parameters(self, ranges_to_skip=None):
    """Returns the next set of parameters to optimise.

    Args:
      ranges_to_skip: Explicitly defines a set of keys to skip.
    """
    if ranges_to_skip is None:
      ranges_to_skip = set(self.results.index)

    if not isinstance(self.param_ranges, dict):
      raise ValueError("Only works for random search!")

    param_range_keys = list(self.param_ranges.keys())
    param_range_keys.sort()

    def _get_next():
      """Returns next hyperparameter set per try."""

      parameters = {
          k: np.random.choice(self.param_ranges[k]) for k in param_range_keys
      }

      # Adds fixed params
      for k in self.fixed_params:
        parameters[k] = self.fixed_params[k]

      return parameters

    for _ in range(self._max_tries):

      parameters = _get_next()
      name = self._get_name(parameters)

      if name not in ranges_to_skip:
        return parameters

    raise ValueError("Exceeded max number of hyperparameter searches!!")

  def update_score(self, parameters, loss, model, info=""):
    """Updates the results from last optimisation run.

    Args:
      parameters: Hyperparameters used in optimisation.
      loss: Validation loss obtained.
      model: Model to serialised if required.
      info: Any ancillary information to tag on to results.

    Returns:
      Boolean flag indicating if the model is the best seen so far.
    """

    if np.isnan(loss):
      loss = np.Inf

    if not os.path.isdir(self.hyperparam_folder):
      os.makedirs(self.hyperparam_folder)

    name = self._get_name(parameters)

    is_optimal = self.results.empty or loss < self.best_score

    # save the first model
    if is_optimal:
      # Try saving first, before updating info
      if model is not None:
        print("Optimal model found, updating")
        model.save(self.hyperparam_folder)
      self.best_score = loss
      self.optimal_name = name

    self.results[name] = pd.Series({"loss": loss, "info": info})
    self.saved_params[name] = pd.Series(parameters)

    self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
    self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

    return is_optimal

class WOAOptManager:
  class WOA:
    def __init__(self,pop,param_ranges,max_iter,loc):
      self._pop = pop
      self._dim = len(param_ranges)
      self._lb = [sorted(x)[0] for x in param_ranges.values()]
      self._ub = [sorted(x)[-1] for x in param_ranges.values()]
      self._max_iter=max_iter
      self._cur_iter=0
      self._pop_iter=0
      # self._a = 0
      # self._a2 = 0
      self._X=self._initial(pop, self._dim, self._ub, self._lb,loc)
      self._fitness=np.zeros((pop, 1))
      self._GbestPositon = np.zeros((1, self._dim))
      self._GbestScore=100
      self._GbestScore_l=[]
      print("Whale optimization algorithm used")
    def getX(self):
      if self._pop_iter==0:
        return self._X
      return None

    def _SortFitness(self,Fit):
      fitness = np.sort(Fit, axis=0)
      index = np.argsort(Fit, axis=0)
      return fitness, index

    def _SortPosition(self,X, index):
      Xnew = np.zeros(X.shape)
      for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
      return Xnew

    def _BorderCheck(self,X, ub, lb, pop, dim):
      for i in range(pop):
        for j in range(dim):
          if X[i, j] > ub[j]:
            X[i, j] = ub[j]
          elif X[i, j] < lb[j]:
            X[i, j] = lb[j]
      return X
    def get_Gbestscore_his(self):
      return self._GbestScore_l
    def update(self):
      fitness, sortIndex = self._SortFitness(self._fitness)
      self._X = self._SortPosition(self._X, sortIndex)
      if fitness[0] < self._GbestScore:
        self._GbestScore = fitness[0]
        self._GbestPositon[0, :]=self._X[0,:]
      self._GbestScore_l.append(self._GbestScore[0])
      if self._cur_iter<self._max_iter/self._pop:
        Leader = self._X[0, :]
        a = 2 - self._cur_iter * (2 / self._max_iter)
        a2 = -1 + self._cur_iter * (-1 / self._max_iter)
        for i in range(self._pop):
          r1 = random.random()
          r2 = random.random()
          A = 2 * a * r1 - a
          C = 2 * r2
          b = 1
          l = (a2 - 1) * random.random() + 1
          for j in range(self._dim):
            p = random.random()
            if p < 0.5:
              if np.abs(A) >= 1:
                rand_leader_index = min(int(np.floor(self._pop * random.random() + 1)), self._pop - 1)
                X_rand = self._X[rand_leader_index, :]
                D_X_rand = np.abs(C * X_rand[j] - self._X[i, j])
                self._X[i, j] = X_rand[j] - A * D_X_rand
              elif np.abs(A) < 1:
                D_Leader = np.abs(C * Leader[j] - self._X[i, j])
                self._X[i, j] = Leader[j] - A * D_Leader
            elif p >= 0.5:
              distance2Leader = np.abs(Leader[j] - self._X[i, j])
              self._X[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + Leader[j]

        print(['iteration' + str(self._cur_iter+1) + ' result' + str(self._GbestScore)])
        print(['iteration' + str(self._cur_iter+1) + ' best param' + str(self._GbestPositon)])
        self._cur_iter = self._cur_iter + 1
        self._X = self._BorderCheck(self._X, self._ub, self._lb, self._pop, self._dim)
    def _initial(self,pop, dim, ub, lb,loc):
      X = np.zeros((pop, dim))
      bound = [lb, ub]
      print(bound)
      for i in range(pop):
        for j in range(dim):
          X[i][j] = round(np.random.uniform(bound[0][j], bound[1][j])) if loc[j] else np.random.uniform(bound[0][j], bound[1][j])
        # X[i] = (X[i,0],round(X[i,1]),round(X[i,2]),X[i,3],X[i,4],round(X[i,5]))
        # X[i]=np.ndarray([round(X[i,k]) for k in range(self._dim) if loc[k]])
      print(X)
      return X
    def getNextPop(self):
      return self._X[self._pop_iter%self._pop,:]
    def getPopIter(self):
      return self._pop_iter
    def getFitness(self,fitness):
      index=self._pop_iter%self._pop
      self._fitness[index]=fitness
      self._pop_iter=self._pop_iter+1
      if(index==self._pop-1):
        # if self._cur_iter==0:
        #   self._GbestPositon[0, :] = self._X[0, :]
        #   self._GbestScore=self._SortFitness(self._fitness)[0][0]
        self.update()
    # def getCurrentBestParam(self):
    #   return self._GbestScore, self._GbestPositon
    def getBestParam(self):
      return self._GbestScore, self._GbestPositon
  def __init__(self,
               param_ranges,
               fixed_params,
               model_folder,
               override_w_fixed_params=True, loc=None):
    """Instantiates model.

    Args:
      param_ranges: Discrete hyperparameter range for Whaleoptimization search.
      fixed_params: Fixed model parameters per experiment.
      model_folder: Folder to store optimisation artifacts.
      override_w_fixed_params: Whether to override serialsed fixed model
        parameters with new supplied values.
    """

    if loc is None:
      loc = [0, 1, 1, 0, 0, 1]
    self.param_ranges = param_ranges
    self._max_tries = 1000
    self._woa_max_iter=10
    self._pop=10
    self.results = pd.DataFrame()
    self.fixed_params = fixed_params
    self.saved_params = pd.DataFrame()
    self.best_score = np.Inf
    self.optimal_name = ""
    self._WOAopt=self.WOA(self._pop,param_ranges,
                         self._pop*self._woa_max_iter,loc)
    # Setup
    # Create folder for saving if its not there
    self.hyperparam_folder = model_folder
    utils.create_folder_if_not_exist(self.hyperparam_folder)

    self._override_w_fixed_params = override_w_fixed_params
  # def set_WOA_param(self,pop):
  #     self.pop=pop
  def getPopandIter(self):
      return self._pop,self._woa_max_iter
  def get_WOA_his(self):
    return self._WOAopt.get_Gbestscore_his()
  def getIter(self):
    return self._WOAopt.getPopIter()
  def getInitialPop(self):
    return self._WOAopt.getX()
  def load_results(self):
    """Loads results from previous hyperparameter optimisation.

    Returns:
      A boolean indicating if previous results can be loaded.
    """
    print("Loading results from", self.hyperparam_folder)

    results_file = os.path.join(self.hyperparam_folder, "results.csv")
    params_file = os.path.join(self.hyperparam_folder, "params.csv")

    if os.path.exists(results_file) and os.path.exists(params_file):

      self.results = pd.read_csv(results_file, index_col=0)
      self.saved_params = pd.read_csv(params_file, index_col=0)

      if not self.results.empty:
        self.results.at["loss"] = self.results.loc["loss"].apply(float)
        self.best_score = self.results.loc["loss"].min()

        is_optimal = self.results.loc["loss"] == self.best_score
        self.optimal_name = self.results.T[is_optimal].index[0]

        return True

    return False

  def _get_params_from_name(self, name):
    """Returns previously saved parameters given a key."""
    params = self.saved_params

    selected_params = dict(params[name])
    # X=self.WOAopt.getX()
    selected_params1=self._WOAopt.getBestParam()[1]
    selected_params={k: selected_params1[0, i] for i, k in enumerate(selected_params.keys()) if k not in self.fixed_params.keys()}
    if self._override_w_fixed_params:
      for k in self.fixed_params:
        selected_params[k] = self.fixed_params[k]

    return selected_params

  def get_best_params(self):
    """Returns the optimal hyperparameters thus far."""

    optimal_name = self.optimal_name

    return self._get_params_from_name(optimal_name)

  def clear(self):
    """Clears all previous results and saved parameters."""
    shutil.rmtree(self.hyperparam_folder)
    os.makedirs(self.hyperparam_folder)
    self.results = pd.DataFrame()
    self.saved_params = pd.DataFrame()

  def _check_params(self, params):
    """Checks that parameter map is properly defined."""

    valid_fields = list(self.param_ranges.keys()) + list(
      self.fixed_params.keys())
    invalid_fields = [k for k in params if k not in valid_fields]
    missing_fields = [k for k in valid_fields if k not in params]

    if invalid_fields:
      raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(
        invalid_fields, valid_fields))
    if missing_fields:
      raise ValueError("Missing Fields Found {} - Valid ones are {}".format(
        missing_fields, valid_fields))

  def _get_name(self, params):
    """Returns a unique key for the supplied set of params."""

    self._check_params(params)

    fields = list(params.keys())
    fields.sort()

    return "_".join([str(params[k]) for k in fields])

  def get_next_parameters(self, ranges_to_skip=None):
    """Returns the next set of parameters to optimise.

    Args:
      ranges_to_skip: Explicitly defines a set of keys to skip.
    """
    if ranges_to_skip is None:
      ranges_to_skip = set(self.results.index)

    if not isinstance(self.param_ranges, dict):
      raise ValueError("Only works for random search!")

    param_range_keys = list(self.param_ranges.keys())
    #param_range_keys.sort()

    def _get_next():
      """Returns next hyperparameter set per try."""

      # parameters = {
      #   k: np.random.choice(self.param_ranges[k]) for k in param_range_keys
      # }
      position=self._WOAopt.getNextPop()
      parameters={k:position[i]  for i, k in enumerate(param_range_keys)}
      # Adds fixed params
      for k in self.fixed_params:
        parameters[k] = self.fixed_params[k]

      return parameters

    for _ in range(self._max_tries):

      parameters = _get_next()
      name = self._get_name(parameters)

      if name not in ranges_to_skip:
        return parameters

    raise ValueError("Exceeded max number of hyperparameter searches!!")

  def update_score(self, parameters, loss, model, info=""):
    """Updates the results from last optimisation run.

    Args:
      parameters: Hyperparameters used in optimisation.
      loss: Validation loss obtained.
      model: Model to serialised if required.
      info: Any ancillary information to tag on to results.

    Returns:
      Boolean flag indicating if the model is the best seen so far.
    """

    if np.isnan(loss):
      loss = np.Inf

    if not os.path.isdir(self.hyperparam_folder):
      os.makedirs(self.hyperparam_folder)

    name = self._get_name(parameters)

    self._WOAopt.getFitness(loss)
    # bestsore,bestposition=self.WOAopt.getCurrentBestParam()
    # X = self.WOAopt.getX()

    is_optimal = self.results.empty or loss < self.best_score #改

    # save the first model
    if is_optimal:
      # Try saving first, before updating info
      if model is not None:
        print("Optimal model found, updating")
        model.save(self.hyperparam_folder)
      self.best_score = loss
      self.optimal_name = name

    self.results[name] = pd.Series({"loss": loss, "info": info})
    self.saved_params[name] = pd.Series(parameters)

    self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
    self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

    return is_optimal


class DistributedHyperparamOptManager(HyperparamOptManager):
  """Manages distributed hyperparameter optimisation across many gpus."""

  def __init__(self,
               param_ranges,
               fixed_params,
               root_model_folder,
               worker_number,
               search_iterations=1000,
               num_iterations_per_worker=5,
               clear_serialised_params=False):
    """Instantiates optimisation manager.

    This hyperparameter optimisation pre-generates #search_iterations
    hyperparameter combinations and serialises them
    at the start. At runtime, each worker goes through their own set of
    parameter ranges. The pregeneration
    allows for multiple workers to run in parallel on different machines without
    resulting in parameter overlaps.

    Args:
      param_ranges: Discrete hyperparameter range for random search.
      fixed_params: Fixed model parameters per experiment.
      root_model_folder: Folder to store optimisation artifacts.
      worker_number: Worker index definining which set of hyperparameters to
        test.
      search_iterations: Maximum numer of random search iterations.
      num_iterations_per_worker: How many iterations are handled per worker.
      clear_serialised_params: Whether to regenerate hyperparameter
        combinations.
    """

    max_workers = int(np.ceil(search_iterations / num_iterations_per_worker))

    # Sanity checks
    if worker_number > max_workers:
      raise ValueError(
          "Worker number ({}) cannot be larger than the total number of workers!"
          .format(max_workers))
    if worker_number > search_iterations:
      raise ValueError(
          "Worker number ({}) cannot be larger than the max search iterations ({})!"
          .format(worker_number, search_iterations))

    print("*** Creating hyperparameter manager for worker {} ***".format(
        worker_number))

    hyperparam_folder = os.path.join(root_model_folder, str(worker_number))
    super().__init__(
        param_ranges,
        fixed_params,
        hyperparam_folder,
        override_w_fixed_params=True)

    serialised_ranges_folder = os.path.join(root_model_folder, "hyperparams")
    if clear_serialised_params:
      print("Regenerating hyperparameter list")
      if os.path.exists(serialised_ranges_folder):
        shutil.rmtree(serialised_ranges_folder)

    utils.create_folder_if_not_exist(serialised_ranges_folder)

    self.serialised_ranges_path = os.path.join(
        serialised_ranges_folder, "ranges_{}.csv".format(search_iterations))
    self.hyperparam_folder = hyperparam_folder  # override
    self.worker_num = worker_number
    self.total_search_iterations = search_iterations
    self.num_iterations_per_worker = num_iterations_per_worker
    self.global_hyperparam_df = self.load_serialised_hyperparam_df()
    self.worker_search_queue = self._get_worker_search_queue()

  @property
  def optimisation_completed(self):
    return False if self.worker_search_queue else True

  def get_next_parameters(self):
    """Returns next dictionary of hyperparameters to optimise."""
    param_name = self.worker_search_queue.pop()

    params = self.global_hyperparam_df.loc[param_name, :].to_dict()

    # Always override!
    for k in self.fixed_params:
      print("Overriding saved {}: {}".format(k, self.fixed_params[k]))

      params[k] = self.fixed_params[k]

    return params

  def load_serialised_hyperparam_df(self):
    """Loads serialsed hyperparameter ranges from file.

    Returns:
      DataFrame containing hyperparameter combinations.
    """
    print("Loading params for {} search iterations form {}".format(
        self.total_search_iterations, self.serialised_ranges_path))

    if os.path.exists(self.serialised_ranges_folder):
      df = pd.read_csv(self.serialised_ranges_path, index_col=0)
    else:
      print("Unable to load - regenerating serach ranges instead")
      df = self.update_serialised_hyperparam_df()

    return df

  def update_serialised_hyperparam_df(self):
    """Regenerates hyperparameter combinations and saves to file.

    Returns:
      DataFrame containing hyperparameter combinations.
    """
    search_df = self._generate_full_hyperparam_df()

    print("Serialising params for {} search iterations to {}".format(
        self.total_search_iterations, self.serialised_ranges_path))

    search_df.to_csv(self.serialised_ranges_path)

    return search_df

  def _generate_full_hyperparam_df(self):
    """Generates actual hyperparameter combinations.

    Returns:
      DataFrame containing hyperparameter combinations.
    """

    np.random.seed(131)  # for reproducibility of hyperparam list

    name_list = []
    param_list = []
    for _ in range(self.total_search_iterations):
      params = super().get_next_parameters(name_list)

      name = self._get_name(params)

      name_list.append(name)
      param_list.append(params)

    full_search_df = pd.DataFrame(param_list, index=name_list)

    return full_search_df

  def clear(self):  # reset when cleared
    """Clears results for hyperparameter manager and resets."""
    super().clear()
    self.worker_search_queue = self._get_worker_search_queue()

  def load_results(self):
    """Load results from file and queue parameter combinations to try.

    Returns:
      Boolean indicating if results were successfully loaded.
    """
    success = super().load_results()

    if success:
      self.worker_search_queue = self._get_worker_search_queue()

    return success

  def _get_worker_search_queue(self):
    """Generates the queue of param combinations for current worker.

    Returns:
      Queue of hyperparameter combinations outstanding.
    """
    global_df = self.assign_worker_numbers(self.global_hyperparam_df)
    worker_df = global_df[global_df["worker"] == self.worker_num]

    left_overs = [s for s in worker_df.index if s not in self.results.columns]

    return Deque(left_overs)

  def assign_worker_numbers(self, df):
    """Updates parameter combinations with the index of the worker used.

    Args:
      df: DataFrame of parameter combinations.

    Returns:
      Updated DataFrame with worker number.
    """
    output = df.copy()

    n = self.total_search_iterations
    batch_size = self.num_iterations_per_worker

    max_worker_num = int(np.ceil(n / batch_size))

    worker_idx = np.concatenate([
        np.tile(i + 1, self.num_iterations_per_worker)
        for i in range(max_worker_num)
    ])

    output["worker"] = worker_idx[:len(output)]

    return output
