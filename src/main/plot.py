#!/usr/bin/env python
__author__ = 'Danelle Cline'
__copyright__ = "Copyright 2020, MBARI"
__license__ = 'GPL v3'
__contact__ = 'dcline at mbari.org'
__doc__ = '''

Plots smoothed mAP from tensorflow run
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import matplotlib

matplotlib.use('Agg')
from pylab import *
import pandas as pd

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 30
import tensorflow as tf


def wall_to_gpu_time(x, zero_time):
    return round(int((x - zero_time) / 60), 0)


def calc_average_precision(x):
    return round(int(x * 100), 0)


def smooth(data, smooth_weight):
    # 1st-order IIR low-pass filter to attenuate the higher-frequency components of the time-series.
    smooth_data = []
    last = 0
    numAccum = 0
    for nextVal in data:
        last = last * smooth_weight + (1 - smooth_weight) * nextVal;
        numAccum += 1
        debiasWeight = 1
        if smooth_weight != 1.0:
            debiasWeight = 1.0 - pow(smooth_weight, numAccum)
        smoothed = last / debiasWeight
        smooth_data.append(smoothed)
    return smooth_data


def plot(df, identifier, figure_path, csv_path):
    df_final = pd.DataFrame()
    if not df.empty:
        time_start = df.wall_time[0]

        # convert wall time and value to rounded values
        df['wall_time'] = df['wall_time'].apply(wall_to_gpu_time, args=(time_start,))
        df['value'] = df['value'].apply(calc_average_precision)

        df_final = df_final.append(df)
        # rename columns to something more aligned with the axis labels
        df_final.columns = ['GPU Time', 'step', 'Overall mAP']

    # drop the step column as it's not needed
    df_final = df_final.drop(['step'], axis=1)

    with plt.style.context('ggplot'):
        # start a new figure - size is in inches
        fig = plt.figure(figsize=(6, 4), dpi=400)
        ax1 = plt.subplot(aspect='equal')
        ax1.set_ylim([0, 100])
        ax1.set_ylabel('mAP')
        ax1.set_xlabel('GPU Time (minutes)')
        ax1.set_title('Mean Average Precision', fontstyle='italic')

        m = '.'
        c = 'grey'
        x = df_final['GPU Time'].values
        y = df_final['Overall mAP'].values
        ax1.scatter(x, y, marker=m, color=c, s=20, label=identifier)

        smoothing_weight = 0.8
        y_smooth = smooth(y, smoothing_weight)
        ax1.plot(x, y_smooth, color=c)

        plt.savefig(figure_path, format='png', bbox_inches='tight')
        df_final.to_csv(csv_path)

    print('Done creating {}'.format(figure_path))


if __name__ == '__main__':
    tf.app.run()
