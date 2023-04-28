import panel as pn
import holoviews as hv
import panel.command
#import numpy
import fick_classes
import pandas as pd
import openpyxl
import os, sys
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
from bokeh.models.formatters import PrintfTickFormatter
import time
import bokeh
import plotly.express as px
from datetime import datetime
import plotly.io as pio
import PySimpleGUI as sg #библиотека для создания шкалы процесса
from PIL import Image, ImageDraw
from turtle import *
pio.templates

pid = os.getpid()
hv.extension('plotly')


radio_group = pn.widgets.Select(
    value='Основные параметры',
    options=['Основные параметры', 'Дополнительные параметры']
)

float_width = pn.widgets.FloatSlider(name='Ширина образцов', start=0.01, end=0.1, step=0.01, value=0.01,
                                     format=PrintfTickFormatter(format='%.2f м'))
float_length = pn.widgets.FloatSlider(name='Длина образцов', start=0.1, end=1, step=0.1, value=0.5,
                                      format=PrintfTickFormatter(format='%.2f м'))

float_height = pn.widgets.FloatSlider(name='Высота образца', start=0.01, end=0.2, step=0.001, value=0.05,
                                      format=PrintfTickFormatter(format='%.3f м'))

float_volume = pn.widgets.FloatSlider(name='Объём аппарата', start=0.1, end=1, step=0.1, value=0.3,
                                      format=PrintfTickFormatter(format='%.2f кубических метров'))

float_flowrate = pn.widgets.FloatSlider(name='Объёмный расход', start=0.000001, end=0.001, step=0.000001, value=0.0001,
                                        format=PrintfTickFormatter(format='%.6f кубических метров в секунду'))

float_dt = pn.widgets.FloatSlider(name='Шаг по времени', start=10, end=200, step=10, value=50)

float_diff_coef = pn.widgets.FloatSlider(name='Коэффициент диффузии', start=0.000000001, end=0.00000001, step=0.000000001,
                         value=0.00000001, format=PrintfTickFormatter(format='%.0e м2/сек'))

int_number_samples = pn.widgets.IntSlider(name='Количество образцов', start=100, end=1000, step=100, value=200,
                                          format=PrintfTickFormatter(format='%.1f штук'))


group_of_key = pn.widgets.RadioButtonGroup(
    name='Выбор необходимого вида образца', options=['one_dim', 'cyl', 'sphere'], button_type='success',
    orientation='vertical')

group_of_ways = pn.widgets.RadioButtonGroup(
    name='Выбор разностной схемы', options=['implicit', 'explicit', 'implicit modified'], button_type='success', orientation='vertical')

groups = pn.Row(group_of_key, group_of_ways)

groups_main = pn.Column(float_width, float_length, float_height, float_volume, float_flowrate, float_dt, float_diff_coef, int_number_samples, groups)


static_cond = pn.widgets.StaticText(name='Условие устойчивости', value=' ')
static_text = pn.widgets.StaticText(name='Поле для вывода ошибок', value=' ')
static_time = pn.widgets.StaticText(name='Время расчёта', value=' ')
static_time_process = pn.widgets.StaticText(name='Расчётное время проведения сушки в часах', value=' ')
static_time_process_result = pn.widgets.StaticText(name='Итог сушки', value=' ')


button = pn.widgets.Button(name='Нажмите для запуска расчёта', button_type='primary')
button_exit = pn.widgets.Button(name='Нажмите для выхода', button_type='primary')


pressure_select = pn.widgets.FloatSlider(name='Давление системы', start=80, end=200, step=5, value=120,
                                          format=PrintfTickFormatter(format='%.1f бар'))
temperature_select = pn.widgets.FloatSlider(name='Температура системы', start=313, end=500, step=1, value=313,
                                          format=PrintfTickFormatter(format='%.1f кельвинов'))

dop_column = pn.Column(pressure_select, temperature_select)

#виджеты для отображения

main_column = pn.WidgetBox('# Расчёт процесса сверхкритической сушки', radio_group, groups_main, dop_column,
                          static_text, static_cond, static_time,
                          static_time_process, static_time_process_result,  button, button_exit)

plot = hv.Curve([0]).opts(width=300)
picture = None
main_window = pn.Row(pn.Spacer(width=100), main_column, pn.Spacer(width=50), plot,pn.Spacer(width=50), picture, sizing_mode='stretch_width') # общий виджет


def visual(volume, height, length, width, type):
    im = Image.new('RGB', (500, 500), (255, 255, 255))

    '''это для квадрата'''
    x0 = 50; y0 = 120; x1 = 350; y1 = 420
    print('Объём', volume)
    width_rect = 10

    side = pow(volume,1/3)
    print('Длина стороны реактора', side)
    equal = side/height
    print('Высота', height, 'отношение сторон',equal)
    diam = int((x1 -x0)/equal)

    """для дуги"""
    x0_arc = 50; y0_arc = 100; x1_arc = 350; y1_arc = 150
    width_arc = 10

    draw = ImageDraw.Draw(im)
    sq = draw.rectangle(xy=(x0, y0, x1, y1), fill='white', outline=(0, 0, 0), width=width_rect)
    draw.arc(xy=(x0_arc, y0_arc, x1_arc, y1_arc), start=180, end=360, fill='black', width=width_arc)

    tests = ['test1', 'test2']
    if type == 'sphere':
        for i in range(len(tests)):
            for g in range(0, y1 - y0 - diam - 2* width_rect , diam):
                for k in range(0, x1 - x0 - diam - 2* width_rect, diam):
                    i = draw.ellipse(xy=(x0 + width_rect + k, y0 + width_rect, x0 + width_rect + diam + k, y0 + width_rect + diam),
                        fill='blue', outline=None, width=3)

                y0 = y0 + diam
    if type == 'cyl':
        for i in range(len(tests)):
            for g in range(0, y1 - y0 - diam - 2* width_rect , diam):
                for k in range(0, x1 - x0 - diam - 2* width_rect, diam):
                    i = draw.ellipse(
                        xy=(x0 + width_rect + k, y0 + width_rect, x0 + width_rect + diam + k, y0 + width_rect + diam),
                        fill='blue', outline=None, width=3)
                    i = draw.line(xy=(x0 + width_rect + k + diam/2, y0 + width_rect , x0 + width_rect + k + diam/2 + 20, y0 + width_rect), fill = 'black')
                    i = draw.line(xy=(x0 + width_rect + k + diam / 2, y0 + width_rect + diam, x0 + width_rect + k + diam / 2 + 20,y0 + width_rect +diam), fill='black')
                    i = draw.arc(xy=(x0 + width_rect + k + diam/2 + 20, y0 + width_rect, x0 + width_rect + k + diam/2 + 20 + 20, y0 + width_rect +diam), start=270, end=90, fill='red', width=3)
                y0 = y0 + diam




    path = os.path.join(r'C:\Users\danko\OneDrive\Рабочий стол\Diplom-python\Fick\Images', 'Version1.jpg')
    im.save(path)
    im.show()
    return im

def onClose(event): # убивает процесс
    static_text.value = 'До новых встреч'
    panel.command.die('By')

def get_condition():
    if cond_scheme == 5:
        static_cond.value = 'Условие устойчивости не выполняется '
    else:
        static_cond.value = 'Условие выполянется'

def get_time_drying(n):
    static_time_process.value = n * float_dt.value / 3600

def view_end_process(slovo):
    static_time_process_result.value = slovo

def work_process():
    if podskazka == 5:
        static_text.value = 'Ошибка. Объём аппарата превышен'
    else:
        static_text.value = 'Всё работает правильно'

def get_time():
    static_time.value = delta_time


def save_time(var_time):
    list_time = []
    list_time.append(var_time)
    return list_time

def show_time_process(list):
    condition_stop = list[0] * 0.05

    for n, value in enumerate(list):
        stage = round(abs(value-list[0])/list[0] * 100/0.95, 1)

        if value <= condition_stop:
            slovo = 'Сушка проведена успешно'
            break
        else:
            slovo = 'Время сушки недостаточно. Содержание спирта в образцах выше запланированного'
    return n, slovo

def run(event):
    start_time = datetime.now()
    global main_window, float_width, float_dt, float_length, float_diff_coef, int_number_samples, podskazka, delta_time, cond_scheme, temperature_select, pressure_select

    matrix_of_c, list_of_mass, c_app, time, i, r_list, podskazka, cond_scheme = fick_classes.main(temperature_select.value, pressure_select.value, float_width.value, float_length.value, float_height.value,
        float_volume.value, float_flowrate.value, float_dt.value, float_diff_coef.value, int_number_samples.value, group_of_key.value, group_of_ways.value, static_text.value,static_cond.value)

    im = visual(float_volume.value, float_height.value, float_length.value, float_width.value, group_of_key.value)

    # file_excel = pd.DataFrame(list_of_mass)
    # file_excel.to_excel('Ver1.xlsx')

    n, slovo= show_time_process(list_of_mass)

    get_time_drying(n)
    view_end_process(slovo)
    template = "plotly_white"

    plot_mass = go.Figure(data = go.Scatter(y = list_of_mass, x = time/3600))
    plot_mass.update_layout(title="График изменения массы", font = dict(family = "Overpass"), height=500,width=500, template = template,
                      xaxis_title= 'Время, ч', yaxis_title='Масса спирта в образцах, кг')
    plot_mass.update_xaxes(gridcolor='Black')
    plot_mass.update_yaxes(gridcolor='Black')

    time_ratio = 100
    plot_conc  = go.Figure()
    for l,n in enumerate(matrix_of_c[::time_ratio]):
        plot_conc.add_trace(go.Scatter(y=n, x = r_list,name=str(l)+' шаг'))
    plot_conc.update_layout(title="График изменения концентрации", font = dict(family = "Overpass"), height=500,width=500,template = template,
                      xaxis_title='Радиус образцов, м', yaxis_title='Концентрация спирта в образцах, кг/м3')
    plot_conc.update_xaxes(gridcolor='Black')
    plot_conc.update_yaxes(gridcolor='Black')

    plot_3d = go.Figure(data=[go.Surface(x=r_list, y=time, z=matrix_of_c)])
    plot_3d.update_layout(title="График отображения 3D концентрации", font = dict(family = "Overpass"), template = template, scene=dict(xaxis_title='Радиус, м',yaxis_title='Время, с',
                zaxis_title='Концентрация спирта, кг/метр3', xaxis = dict(gridcolor='LightPink'), yaxis = dict(gridcolor='LightPink'), zaxis = dict(gridcolor='LightPink')),
                          width=500, height=500, margin=dict(l=10, r=20, b=35, t=30))


    #plot_time_and_P_T = go.Figure(data = [go.Surface(x = P, y = T, z = hhh)])


    get_condition()
    work_process()
    plots = pn.Row(pn.Spacer(width=100), plot_mass, pn.Spacer(width=50), plot_conc, pn.Spacer(width=50), sizing_mode='stretch_width')
    plot_3d_main = pn.Row(pn.Spacer(width = 100), plot_3d, pn.Spacer( width = 50))
    main_plots = pn.Column(plots, plot_3d_main,  sizing_mode='stretch_width')
    main_window[0] = pn.Spacer (width = 10)
    main_window[1] = main_column
    main_window[2] = pn.Spacer (width = 10)
    main_window[3] = main_plots
    main_window[4] = pn.Spacer(width=10)
    main_window[5] = im

    variable_time = static_time_process.value
    hhh =  save_time(variable_time)
    print('hhh', hhh)
    end_time = datetime.now()
    delta_time = end_time - start_time
    get_time()

def main():
    button.on_click(run)
    button_exit.on_click(onClose)
    main_window.show()

main()