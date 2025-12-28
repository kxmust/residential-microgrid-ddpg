import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os

# 导入数据
load_data = pd.read_csv(r'data/home_1_training_30days.csv',
                        header=0).iloc[:, 1]
pv_data = pd.read_csv(r'data/home_1_training_30days.csv',
                      header=0).iloc[:, 2]
price_data = pd.read_csv(r'data/electricity_price_training.csv',
                         header=0).iloc[:, 1]

class RM_Data:
    def __init__(self, load_data, pv_data, price_data):
        self.pv_data = pv_data
        self.load_data = load_data
        self.price_data = price_data
        self.len_data = len(load_data)

    def pv(self, t):
        return self.pv_data[t]

    def ug_price(self, t):
        return self.price_data[t]

    def load(self, t):
        return self.load_data[t]

    def plot_figure(self, data_type, hours, save_dir='plots'):
        """
        绘制时间序列数据图表

        参数:
        data_type: 数据类型，可选 'pv'、'load'、'price'
        hours: 时间长度（小时）
        save_dir: 图片保存目录
        """

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 生成时间序列（假设从0点开始）
        time_points = list(range(hours))

        # 根据数据类型读取数据
        data = []
        for t in time_points:
            if data_type.lower() == 'pv':
                data.append(self.pv(t))  # 调用pv函数
            elif data_type.lower() == 'load':
                data.append(self.load(t))  # 调用load函数
            elif data_type.lower() == 'price':
                data.append(self.ug_price(t))  # 调用price函数
            else:
                raise ValueError(f"不支持的数据类型: {data_type}，请使用 'pv', 'load' 或 'price'")

        data = np.array(data)

        # 创建时间标签（从当前时间开始）
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        time_stamps = [base_time + timedelta(hours=t) for t in time_points]

        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))

        # 根据数据类型设置图表属性
        if data_type.lower() == 'pv':
            title = f'PV Power (0-{hours}Hours)'
            ylabel = 'Power (kW)'
            color = '#FFA500'  # 橙色
            line_style = '-o'
            alpha = 0.7
        elif data_type.lower() == 'load':
            title = f'Load Power (0-{hours}Hours)'
            ylabel = 'Power (kW)'
            color = '#1E90FF'  # 道奇蓝
            line_style = '-s'
            alpha = 0.7
        elif data_type.lower() == 'price':
            title = f'Electricity Price (0-{hours}Hours)'
            ylabel = 'Price (yuan/kWh)'
            color = '#32CD32'  # 石灰绿
            line_style = '-^'
            alpha = 0.7

        # 绘制线图
        ax.plot(time_stamps, data, line_style, color=color,
                linewidth=2, markersize=6, alpha=alpha,
                label=data_type.upper())

        # 填充区域（针对pv和load）
        if data_type.lower() in ['pv', 'load']:
            ax.fill_between(time_stamps, 0, data, alpha=0.2, color=color)

        # 设置图表属性
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # 设置x轴时间格式
        if hours <= 24:
            # 每小时显示一个标签
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours // 12)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            # 超过24小时，按天显示
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))

        # 旋转x轴标签
        plt.xticks(rotation=45)

        # 添加网格
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.5)

        # 添加图例
        ax.legend(loc='best', fontsize=10)

        # 添加统计信息文本框
        stats_text = f"""
        Max Value: {data.max():.2f}
        Min Value: {data.min():.2f}
        Avg Value: {data.mean():.2f}
        Total Times: {hours}Hours
        """

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 自动调整布局
        plt.tight_layout()

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{hours}h_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)

        # 保存图片
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"The figure is saved: {filepath}")

        # 显示图表
        plt.show()

        return filepath

rm_data = RM_Data(load_data, pv_data, price_data)
# print(rm_data.len_data)
# rm_data.plot_figure('price', hours=24)
# rm_data.plot_figure('load', hours=24)
# rm_data.plot_figure('pv', hours=24)




