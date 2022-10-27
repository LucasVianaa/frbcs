import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math


sns.set(style="darkgrid")
data = {
        'height': [],
        'length': [],
        'area': [],
        'eccen': [],
        'p_black': [],
        'p_and': [],
        'mean_tr': [],
        'blackpix': [],
        'blackand': [],
        'wb_trans': [],
        'class': []
    }
data_low = {
        'height': [],
        'length': [],
        'area': [],
        'eccen': [],
        'p_black': [],
        'p_and': [],
        'mean_tr': [],
        'blackpix': [],
        'blackand': [],
        'wb_trans': [],
        'class': []
    }

data_medium = {
        'height': [],
        'length': [],
        'area': [],
        'eccen': [],
        'p_black': [],
        'p_and': [],
        'mean_tr': [],
        'blackpix': [],
        'blackand': [],
        'wb_trans': [],
        'class': []
    }

data_high = {
        'height': [],
        'length': [],
        'area': [],
        'eccen': [],
        'p_black': [],
        'p_and': [],
        'mean_tr': [],
        'blackpix': [],
        'blackand': [],
        'wb_trans': [],
        'class': []
    }



with open("page-blocks.data", "r") as data_file:
    for line in data_file.readlines():
        line_data = line.split()
        data['height'].append(int(line_data[0]))
        data['length'].append(int(line_data[1]))
        data['area'].append(int(line_data[2]))
        data['eccen'].append(float(line_data[3]))
        data['p_black'].append(float(line_data[4]))
        data['p_and'].append(float(line_data[5]))
        data['mean_tr'].append(float(line_data[6]))
        data['blackpix'].append(int(line_data[7]))
        data['blackand'].append(int(line_data[8]))
        data['wb_trans'].append(int(line_data[9]))
        if(int(line_data[10]) == 1):
            data['class'].append("Texto")
        else:
            data['class'].append("Não-texto")


df = pd.DataFrame(data=data)
df1 = df[df['class'] == "Texto"]
df1 = df1.sample(n = 560)
df2 = df[df['class'] == "Não-texto"]

df3 = pd.concat([df1, df2])


for key in data:
    if key != 'class':
        plot = sns.kdeplot(data=df3, x=key, hue="class", log_scale=True)
        print("- "+key)
        print("* texto")
        # Low
        print("> Low")
        print(df1[key].quantile(0), df1[key].quantile(0.01), df1[key].quantile(0.05), "start")
        print(df1[key].quantile(0.95), df1[key].quantile(0.99), df1[key].quantile(1), "end")
        

        # Medium
        print("> Medium")
        print(df1[key].quantile(0.04), df1[key].quantile(0.12), df1[key].quantile(0.20), "medium")
        print(df1[key].quantile(0.65), df1[key].quantile(0.77), df1[key].quantile(0.93), "medium")

        # High
        print("> High")
        print(df1[key].quantile(0.18), df1[key].quantile(0.50), df1[key].quantile(0.70), "high")

        print("* non-texto")
        # Low
        print("> Low")
        print(df2[key].quantile(0), df2[key].quantile(0.01), df2[key].quantile(0.05), "start")
        print(df2[key].quantile(0.95), df2[key].quantile(0.99), df2[key].quantile(1), "end")

        # Medium
        print("> Medium")
        print(df2[key].quantile(0.04), df2[key].quantile(0.12), df2[key].quantile(0.20), "medium")
        print(df2[key].quantile(0.65), df2[key].quantile(0.77), df2[key].quantile(0.93), "medium")

        # High
        print("> High")
        print(df2[key].quantile(0.18), df2[key].quantile(0.50), df2[key].quantile(0.70), "high")

        fig = plot.get_figure()
        fig.savefig("histograms/"+key+".png")
        plt.clf()