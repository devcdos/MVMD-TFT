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

"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import argparse
import datetime as dte
import os
from math import sqrt

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import libs.numpyloss as loss
import numpy as np
import pandas as pd

import tensorflow._api.v2.compat.v1 as tf

from LSTMselfbuild.testLSTM import generate_dir
from libs import plot_fig

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes
#

# tf.compat.v1.disable_eager_execution()
ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
import sys
np.set_printoptions(threshold=sys.maxsize)


# ModelClass=libs.LS
def individual_loss(datamap):
    pass
def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):
  """Trains tft based on defined model params.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    model_folder: Folder path where models are serialized
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
    use_testing_mode: Uses a smaller models and data sizes for testing purposes
      only -- switch to False to use original default settings
  """

  num_repeats = 1

  if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))

  # Tensorflow setup
  default_keras_session = tf.compat.v1.keras.backend.get_session()

  if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

  else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

  print("*** Training from defined parameters for {} ***".format(expt_name))

  print("Loading & splitting data...")
  raw_data = pd.read_csv(data_csv_path, index_col=0)
  # elec_col=raw_data[['categorical_id']].groupby('categorical_id').first().index[:100][-1]
  # raw_data=raw_data[raw_data['categorical_id']<elec_col]
  #
  raw_data=utils.add_copy_col("id",raw_data,"building_id")
  if name=='Huelonger':
      train, valid, test = data_formatter.split_data(raw_data,392 * 24, 441 * 24)
  else:train, valid, test = data_formatter.split_data(raw_data)
  # train, valid, test = data_formatter.split_data(raw_data,72+1096,81+1096)


  # train_samples, valid_samples ,test_samples= data_formatter.get_num_samples_for_calibration()
  #train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

  # vmdcutpath="/".join(data_csv_path.split("\\")[:-1])+"/27/2mode/"
  # vmdcutpath = "./outputs/data/MVMDHue/25/3mode/"
  # vmdcutpath = "./outputs/data/MVMDHue/25/9mode/"
  #vmdcutpath = "./outputs/data/MVMDHue/25/9mode/Nocross/"
  #
  #train, valid, test = data_formatter.set_data(utils.add_copy_col("id",pd.read_csv(vmdcutpath+"train.csv"),"building_id"),
  #                                              utils.add_copy_col("id",pd.read_csv(vmdcutpath+"valid.csv"),"building_id"),
  #                                             utils.add_copy_col("id",pd.read_csv(vmdcutpath+"test.csv"),"building_id"))
  train_samples, valid_samples,test_samples=(data_formatter.get_num_samples_for_calibration(i) for i in (train,valid,test))
  # train_samples, valid_samples,test_samples=data_formatter.get_num_samples_for_calibration()
  # print("trainsamples",train_samples)#测试用例对不上
  # print("valid_samples",valid_samples)
  #
  # Sets up default params
  fixed_params = data_formatter.get_experiment_params()
  params = data_formatter.get_default_model_params()
  params["model_folder"] = model_folder

  # Parameter overrides for testing only! Small sizes used to speed up script.
  if use_testing_mode:
    fixed_params["num_epochs"] = 1
    params["hidden_layer_size"] = 5
    train_samples, valid_samples = 100, 10

  # Sets up hyperparam manager
  print("*** Loading hyperparm manager ***")
  opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                     fixed_params, model_folder)

  # Training -- one iteration only
  print("*** Running calibration ***")
  print("Params Selected:")
  for k in params:
    print("{}: {}".format(k, params[k]))

  best_loss = np.Inf

  def extract_numerical_data(data):
      """Strips out forecast time and identifier columns."""
      return data[[
          col for col in data.columns
          if col not in {"forecast_time", "identifier"}
      ]]

  temp = '' if name == 'Hue' else 'longer/'
  for _ in range(num_repeats):

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

      tf.keras.backend.set_session(sess)

      params = opt_manager.get_next_parameters()
      model = ModelClass(params, use_cudnn=use_gpu)
      if not model.training_data_cached():
          model.cache_batched_data(train, "train", num_samples=train_samples)
          model.cache_batched_data(valid, "valid", num_samples=valid_samples)
      sess.run(tf.global_variables_initializer())
      model.fit()
      combined=utils.individualforecast(model,test,None)#MVMDTFT重新训练
      # # xxxxxxxxx=
      prep50=data_formatter.inverse_transform(combined['predictp50'])
      actual=data_formatter.inverse_transform(combined['actual'])
      prep90 = data_formatter.inverse_transform(combined['predictp90'])
      prep10 = data_formatter.inverse_transform(combined['predictp10'])

      output_map, res, y_origin = model.predict(test, return_targets=True)


      targets = data_formatter.format_predictions(output_map["targets"])
      print(targets)
      L=10607 if name=='Huelonger' else 1967
      p90_forecast = data_formatter.format_predictions(output_map["p90"])  ##P50 和P90是啥
      p10_forecast= data_formatter.format_predictions(output_map["p10"])
      p50_forecast = data_formatter.format_predictions(output_map["p50"])
      targetsF = targets[
                         targets["forecast_time"] >= L
                         ]
      # targetsF = targets[(targets['identifier'] !="Residential_6")&
      #                    (targets['identifier'] != "Residential_10")&
      #                    (targets['identifier'] != "Residential_14")&
      #                    (targets["forecast_time"] >= L)
      #                   ]
      p50_forecastF = p50_forecast[
                                   p50_forecast["forecast_time"] >= L]
      p90_forecastF = p90_forecast[p90_forecast["forecast_time"] >= L]
      p10_forecastF= p10_forecast[p10_forecast["forecast_time"] >= L]
      p50_group = p50_forecastF.groupby("identifier")
      p90_group = p90_forecastF.groupby("identifier")
      target_group = targetsF.groupby("identifier")
      p50_loss_l=[]
      p90_loss_l=[]
      l=1152 if name=='Huelonger' else 192
      for i in range(l):
          p50,_,_,_=utils.numpy_normalised_quantile_loss(target_group.nth(i)['t+0'].reset_index(drop=True),
                                              p50_group.nth(i)['t+0'].reset_index(drop=True), 0.5)
          p50_loss_l.append(p50)
          p90, _, _,_ = utils.numpy_normalised_quantile_loss(target_group.nth(i)['t+0'].reset_index(drop=True),
                                                           p90_group.nth(i)['t+0'].reset_index(drop=True), 0.9)
          p90_loss_l.append(p90)
      # avg_p50_loss=sum([utils.numpy_normalised_quantile_loss(p50_group.nth(i),target_group.nth(i),0.5) for i in range(1152)])/1152
      # avg_p90_loss=sum([utils.numpy_normalised_quantile_loss(p90_group.nth(i),target_group.nth(i),0.9) for i in range(1152)])/1152
      print(p50_loss_l,p90_loss_l)
      print(sum(p50_loss_l)/l, sum(p90_loss_l)/l)

      #
      # p50_loss=utils.tensorflow_quantile_loss(extract_numerical_data(targets), extract_numerical_data(p50_forecast),
      #       0.5)
      # p90_loss = utils.numpy_normalised_quantile_loss(
      #     extract_numerical_data(targets), extract_numerical_data(p90_forecast),
      #     0.9)
      # p50 = loss.MAE(extract_numerical_data(targets), extract_numerical_data(p50_forecast))
      # p90 = loss.MAE(extract_numerical_data(targets), extract_numerical_data(p90_forecast))
      # print(p50_loss, p90_loss)
      # R20=targetsF[targetsF['identifier']=='Residential_20']

      for b in prep50.keys():
        # if b!="Residential_25":continue
        p50=p50_forecastF[p50_forecastF['identifier']==b]['t+0'].values
        p90=p90_forecastF[p90_forecastF['identifier']==b]['t+0'].values
        p10=p10_forecastF[p10_forecastF['identifier']==b]['t+0'].values

        target=targetsF[targetsF['identifier']==b]['t+0'].values
        # print(R20)

        with open('./bestmodel/TFT/Minmax/'+temp+'quantileresrecord/' + b + '.txt', 'w') as f:
            f.writelines(b + "pre1actual2")
            f.writelines(str(p10))
            f.writelines(str(p90))
            # f.writelines(str(prep10[b][:, 0]))
            # f.writelines(str(prep90[b][:, 0]))
        with open('./bestmodel/TFT/Minmax/'+temp+'resrecord/' + b + '.txt', 'w') as f:
            f.writelines(b+"pre1actual2")
            f.writelines(str(p50))
            f.writelines(str(target))
            # f.writelines(str(prep50[b][:,0]))
            # f.writelines(str(actual[b][:,0]))
        with open('./bestmodel/TFT/Minmax/'+temp+'lossrecord/' + b + '.txt', 'w') as f:
            f.write(b + ":\nMAE:{}\nMSE:{}\nRMSE:{}\nWAPE:"
                          "{}\n".format(
                  # loss.MAE(actual[b],prep50[b]),loss.MSE(actual[b],prep50[b]),sqrt(loss.MSE(actual[b],prep50[b])),loss.WAPE(actual[b],prep50[b])))
                loss.MAE(target, p50), loss.MSE(target, p50), sqrt(loss.MSE(target, p50)),
                loss.WAPE(target, p50)))

if __name__ == "__main__":
  seed = 114514
  def get_args():
    """Gets settings from command line."""

    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "expt_name",
        metavar="e",
        type=str,
        nargs="?",
        default="buildingBGD2",
        choices=experiment_names,
        help="Experiment Name. Default={}".format(",".join(experiment_names)))
    parser.add_argument(
        "output_folder",
        metavar="f",
        type=str,
        nargs="?",
        default=".",
        help="Path to folder for data download")
    parser.add_argument(
        "use_gpu",
        metavar="g",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="yes",
        help="Whether to use gpu for training.")

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder
    args.expt_name="Huelonger"
    #args.expt_name = "electricity"
    return args.expt_name, root_folder, args.use_gpu == "yes"

  name, output_folder, use_tensorflow_with_gpu = get_args()

  print("Using output folder {}".format(output_folder))

  config = ExperimentConfig(name, output_folder)
  formatter = config.make_data_formatter()

  # Customise inputs to mai(n() for new datasets.
  main(#debug quantile loss(p50,p90) how many timesteps predicting?)
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      model_folder=os.path.join(config.model_folder, "fixed"),
      data_csv_path=config.data_csv_path,
      data_formatter=formatter,
      use_testing_mode=False)  # Change to false to use original default params
