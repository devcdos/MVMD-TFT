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

"""Custom formatting functions for Electricity dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes
import numpy as np
class MyMinMax:
    def __init__(self):
        self._min=None
        self._max=None
    def myMinMax(self,data):
        self._min=data.min()
        self._max=data.max()
        data=np.apply_along_axis(lambda x:(x-self._min)/(self._max-self._min),axis=0,arr=data)
        return data
    def inverse(self,data):#inversetransform结果不对
        return np.apply_along_axis(lambda x:x*(self._max-self._min)+self._min,axis=0,arr=data)
class HueFormatter(GenericDataFormatter):
  """Defines and formats data for the electricity dataset.

  Note that per-entity z-score normalization is used here, and is implemented
  across functions.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      #hour            weekend    holiday    dst        weather

      ('id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('energy_kWh', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      #('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      # ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
      ('temperature', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ("humidity", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('pressure', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('dayofmonth', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('weekend', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('building_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('houseType', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('RUs', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('facing', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

  ]
  # _column_definition = [
  #     # hour            weekend    holiday    dst        weather
  #
  #     ('id', DataTypes.REAL_VALUED, InputTypes.ID),
  #     ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
  #     ('energy_kWh', DataTypes.REAL_VALUED, InputTypes.TARGET),
  #     ('IMF1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     ('IMF2', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     # ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  #     # ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),#离散变量还是连续变量
  #     ('temperature', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     ("humidity", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     ('pressure', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     ('dayofmonth', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  #     ('building_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  #     ('day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  # ]
  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']

  def split_data(self, df, valid_boundary=72*24, test_boundary=81*24,diffday=24):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')
    diffday=self.get_fixed_params()['total_time_steps']


    # Format real scalers
    # column_definitions = self.get_column_definition()
    # id_column = utils.get_single_col_by_input_type(InputTypes.ID,
    #                                                column_definitions)
    # inputs = utils.extract_cols_from_data_type(
    #     DataTypes.REAL_VALUED, column_definitions,
    #     {InputTypes.ID, InputTypes.TIME})
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # for (i,d) in list(enumerate(df.groupby(id_column))):
    #     sns.heatmap(d[1][inputs].corr(),
    #             annot=True,  # 显示相关系数的数据
    #             center=0.5,  # 居中
    #             fmt='.2f',  # 只显示两位小数
    #             linewidth=0.5,  # 设置每个单元格的距离
    #             linecolor='blue',  # 设置间距线的颜色
    #             vmin=0, vmax=1,  # 设置数值最小值和最大值
    #             xticklabels=True, yticklabels=True,  # 显示x轴和y轴
    #             square=True,  # 每个方格都是正方形
    #             cbar=True,  # 绘制颜色条
    #             cmap='coolwarm_r',  # 设置热力图颜色
    #             )
    #     plt.show()  # 显示图片
    #index = df['days_from_start']
    index = df['hours_from_start']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary - diffday) & (index < test_boundary)] #index为天数 为啥要-7
    test = df.loc[index >= test_boundary - diffday]

    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME,InputTypes.TARGET})
    # Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):

      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier] \
      = sklearn.preprocessing.MinMaxScaler().fit(data)

        self._target_scaler[identifier] \
      = sklearn.preprocessing.MinMaxScaler().fit(targets)
        # M=MyMinMax()
        # trans1=M.myMinMax(targets)
        # inv=M.inverse(trans1)
        # trans=self._target_scaler[identifier].transform(targets)
        # test = self._target_scaler[identifier].inverse_transform(trans)
      identifiers.append(identifier)

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()#调用父类方法
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME,InputTypes.TARGET})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    target = [x for x in utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME}) if x not in real_inputs]
    # Transform real inputs per entity
    df_list = []
    for identifier, sliced in df.groupby(id_col):

      # Filter out any trajectories that are too short
      if len(sliced) >= self._time_steps:
        sliced_copy = sliced.copy()
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        sliced_copy[target] = self._target_scaler[identifier].transform(
            sliced_copy[target].values)
        df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      target_scaler = self._target_scaler[identifier]

      for col in column_names:
        if col not in {'forecast_time', 'identifier'}:
            #sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
            sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col].values.reshape(-1,1))
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps':25,
        'num_encoder_steps':24,
        #'total_time_steps': 2,
        #'num_encoder_steps': 24,
        'num_epochs':50,
        #'num_epochs': 5,
        'early_stopping_patience': 5,
        #'early_stopping_patience': 2,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""
    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 20,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
    }
    return model_params

  def get_num_samples_for_calibration(self,data=None):

    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    # return 1000

    if data is not None:
        id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                    self.get_column_definition())
        timestep = self.get_fixed_params()['total_time_steps']
        return len([i for i in range(len(data[data[id_col] == data[id_col].iloc[0]]) - timestep )]) * len(
            data.groupby(id_col))
    #return 22021, 3205, 0  # en144 de 12 t0.8,v0.1
    # return 22357, 3541, 0  # en120 de 12 t0.8,v0.1
    #return 22693, 3877, 0  # en96 de 12 t0.8,v0.1
    # return 23029, 4213, 0  # en72 de 12 t0.8,v0.1
    #return 23365, 4549, 0  # en48 de 12 t0.8,v0.1
    # return 22105, 3289, 0  # en144 de 6 t0.8,v0.1
    # return 22441, 3625, 0  # en120 de 6 t0.8,v0.1
    # return 22777, 3961, 0  # en96 de 6 t0.8,v0.1
    # return 23113, 4297, 0  # en72 de 6 t0.8,v0.1
    # return 23449, 4633, 0  # en48 de 6 t0.8,v0.1
    # return 22147, 3331, 0  # en144 de 3 t0.8,v0.1
    # return 22483, 3667, 0  # en120 de 3 t0.8,v0.1
    # return 22819, 4003, 0  # en96 de 3 t0.8,v0.1

    #return 23155, 4339, 0  # en72 de 3 t0.8,v0.1
    # return 23491, 4675, 0  # en48 de 3 t0.8,v0.1
    #return 23827, 5011, 0  # en24 de 3 t0.8,v0.1
    # return 22511, 3695, 0  # en120 de 1 t0.8,v0.1
    # return 22847, 4031, 0  # en96 de 1 t0.8,v0.1
    # return 23183, 4367, 0  # en72 de 1 t0.8,v0.1
    # return 23519, 4703, 0  # en48 de 1 t0.8,v0.1
    # return 23855, 5039, 0  # en24 de 1 t0.8,v0.1

    #return 22357,3541,0  #en120 de 12 t0.8,v0.1
    #return 21685, 2869, 0  # en168 de 12 t0.8,v0.1
    #return 20831,8063,0  #en24 de 1 t0.7,v0.2
    #return 1000, 200,0
  # def get_column_definition(self):
  #     return self._column_definition
  def inverse_transform(self,data):
      entity=data.keys()
      for e in entity:
          data[e]=self._target_scaler[e].inverse_transform(data[e].reshape(-1,1))
      return data