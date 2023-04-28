import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import math
from scipy import optimize
from scipy.optimize import minimize
import time
import PySimpleGUI as psg #библиотека для создания шкалы процесса
from tqdm import tqdm

load_perc = 0.7 # доля загрузки аппарата СКС

target_material_porosity = 120 #кг/м3
porosity = 0.95
tau_izv = 11 #5.5 #извилистость пор

Vcr_ips = 220 #см3/моль - критический молярный объём ips
Vcr_co2 = 94.07
Pcr_ips = 47.62 #критическаое давлние ипс бар
Pcr_co2 = 73.74

PSI_ips = 0.665 #ацентрический фактор ипс
PSI_co2 = 0.225

Par_ips = 164.4 #парахора ипс
Par_co2 = 44.8

ips_volume_norm = 0.285*Vcr_ips**1.048 #нормальный молярынй объём ипс
co2_volume_norm = 0.285*Vcr_co2**1.048

M_ips = 60.09 #г на см3
M_co2 = 44.01

Tcr_ips = 508.3 #критическая температура ипс
Tcr_co2 = 304.12

a1 = 8.07652E+15
a2 = 8.91648E+13
a3 = -2.89429E+13
a4 = 89749140000
a5 = 230022.4

B1 = -1.146067e-1
B2 = 6.978380e-7
B3 = 3.976765e-10
B4 = 6.336120e-2
B5 = -1.166119e-2
B6 = 7.142596e-4
B7 = 6.519333e-6
B8 = -3.567559e-1
B9 = 3.180473e-2


density_sio2 = 2034 #кг/м3
density_ips = 785.1  # кг/м3
density_co2 = 468  # кг/м3


file_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(file_path, 'Images')

num_steps = 100  # количество шагов
l = np.empty(num_steps + 2, dtype=np.int16)

proc_time = 45*36
c_bound = 0.



class scd_apparatus():
    def __init__(self, T, P, volume, flowrate, width, length, height, diff_coef, key, key_sch, number_samples):
        self.width = width
        self.length = length
        self.height = height
        self.diff_coef = diff_coef
        self.key = key
        self.key_sch = key_sch
        self.number_samples = number_samples
        self.c_init_list = np.zeros(num_steps + 2)
        self.matrix_of_c = np.zeros((n_t, len(self.c_init_list)))
        self.list_of_mass = np.zeros(n_t)
        self.c_app = np.zeros(n_t)
        self.volume = volume
        self.flowrate =flowrate
        self.T = T
        self.P = P

        if self.key == 'one_dim':
            self.V_gels = self.height * self.width * self.length * self.number_samples

        elif self.key == 'cyl':
            self.V_gels = self.number_samples * self.length * np.pi * R ** 2

        elif self.key == 'sphere':
            self.V_gels = self.number_samples * 4 / 3 * np.pi * (R) ** 3

        self.V_apparat_free = self.volume - self.V_gels #объём свободного пространства аппарата
        self.m_ips_gramms = self.V_gels * target_material_porosity * density_ips * 1000 #масса ипс

        self.V_total = self.V_apparat_free + self.V_gels * target_material_porosity
    def __str__(self):
        print(
            f'width: {self.width}, length: {self.length}, diff_coef: {self.diff_coef}, number_samples: {self.number_samples}')
        print('Объём гелей', self.V_gels)

    """
    это функция для нахождения истинной плотности улекислого газа в сверхкритическом состоянии и коэффициентов диффузии обоих веществ
    """
    def density_co2_and_viscos(self, T, P):
        density_co2_coef = 14258.88822 - 84.97903074 * self.T + 11.30377033 * self.P + 0.017974145 * self.T * self.P + 0.135119422 * self.T ** 2 - 0.071358164 * self.P ** 2 - 4.73474E-05 * self.T ** 3 + 0.000110024 * self.P ** 3 #плотность сверхкритического углекислого газа
        koef = Tcr_co2 * Vcr_co2 / (1000 * M_co2)
        A_coef = 14.882 + 5.908 * koef + 2.0821 * koef ** 2
        Vmol_co2 = M_co2 / density_co2_coef * 1000  # Молярный объем диоксида углерода при норм температуре кипения, см3/моль
        Vv_co2 = Vmol_co2 / Vcr_co2

        D_coef_ips_co2 = A_coef * 10 ** (-9) * (T / M_ips) ** 0.5 * math.exp(
            -0.3887 / (Vv_co2 - 0.23))  # коэффициент диффузии ипс в CO2

        dynamic_viscosity_ips = (a1 + a2 * P / 10) / (a3 + a4 * T + a5 * T ** 3 + P / 10) / 1000000

        D_coef_co2_ips = 8.93 * 10 ** (-12) * (co2_volume_norm / ips_volume_norm ** 2) ** (1 / 6) * (
                    Par_ips / Par_co2) ** 0.6 * T / (dynamic_viscosity_ips * 1000)

        dynamic_viscosity_co2 = ((B1 + B2 * P + B3 * P ** 2 + B4 * math.log1p(T) + B5 * (math.log1p(T)) ** 2 + B6 * (
            math.log1p(T)) ** 3)/ (1 + B7 * P + B8 * math.log1p(T) + B9 * (math.log1p(T)) ** 2)) / 1000  # Результат в Па*с
        return density_co2_coef , D_coef_ips_co2, D_coef_co2_ips, dynamic_viscosity_ips,  dynamic_viscosity_co2

    """
    эта функция отвечает за создание коэффициента диффузии в зависимости от массовой доли
    """
    def golden_method(self, x):
        numerator = abs(
            (self.V_apparat_free + self.V_gels * target_material_porosity) *
            ((x / density_ips + (1 - x) / density_co2) ** -1) * x
            - (self.m_ips_gramms / 1000))
        return numerator / (self.m_ips_gramms / 1000)

    def fick_conc(self,T, P, c, c_bound, dr, dt, r):
        global sverka_method
        sverka_method = 0
        stab_cond = dt / dr ** 2  # условие устойчивости
        alfa_i = np.zeros(num_steps + 2)
        betta_i = np.zeros(num_steps + 2)
        alfa_i[0] = 1  # прогоночный коэффициент альфа на нулевом шаге
        betta_i[0] = 0  # прогоночный коэффициент бетта на нулевом шаге
        c_temp = []
        c_temp = np.copy(c)

        if self.key == 'one_dim':
            if self.key_sch == 'explicit':
                if stab_cond > 1 / (2 * self.diff_coef):
                    print('stab_cond', stab_cond, 'self.diff', 1 / (2*self.diff_coef))
                    sverka_method = 5
                    pass
                else:
                    c_temp[1:-1] = c[1:-1] + self.diff_coef * (dt / dr ** 2) * (c[2:] - 2 * c[1:-1] + c[0: -2])
                    c_temp[-1] = c_bound
                    c_temp[0] = c_temp[1]
                    return c_temp

            elif self.key_sch == 'implicit':
                a = -self.diff_coef * dt / (dr) ** 2  # коэффициент 'a'
                b = 1 + 2 * self.diff_coef * dt / (dr) ** 2  # коэффицент b
                c_koef = -self.diff_coef * dt / (dr) ** 2  # коэффициент c

                for i in range(1, len(l)):
                    alfa_i[i] = (-a) / (b + c_koef * alfa_i[i - 1])
                    betta_i[i] = (c[i] - c_koef * betta_i[i - 1]) / (b + c_koef * alfa_i[i - 1])
                alfa_i[-1] = 0
                betta_i[-1] = 0
                c_temp[1:-1] = c[2:] * alfa_i[1:-1] + betta_i[1:-1]

                c_temp[-1] = c_bound
                c_temp[0] = c_temp[1]
                return c_temp

        elif self.key == 'cyl':
            if self.key_sch == 'explicit':
                if dt> dr/(2 *self.diff_coef*porosity/tau_izv):
                    sverka_method = 5
                    pass
                else:
                    for i in range(1, len(r) - 1):
                        c_temp[1:-1] = c[1:-1] + self.diff_coef * dt * (
                                    (c[2:] - 2 * c[1:-1] + c[0:-2]) / dr ** 2 + 1 / r[i] * (c[1:-1] - c[0:-2]) / dr)

                    c_temp[-1] = c_bound
                    c_temp[0] = c_temp[1]
                    return c_temp

            elif self.key_sch == 'implicit':  # должна быть абсолютно устойчива это с ЛКР

                for j in range(1, len(r)):
                    a_coef = -self.diff_coef * dt / (dr) ** 2
                    b_coef = 1 + 2 * dt * self.diff_coef / (dr) ** 2 - dt * self.diff_coef / (dr * r[j])
                    c_koef = - dt * self.diff_coef / (dr) ** 2 + dt * self.diff_coef / (dr * r[j])

                for i in range(1, len(l) - 1):
                    alfa_i[i] = (-a_coef) / (b_coef + c_koef * alfa_i[i - 1])
                    betta_i[i] = (c[i] - c_koef * betta_i[i - 1]) / (b_coef + c_koef * alfa_i[i - 1])


                alfa_i[-1] = 0
                betta_i[-1] = 0
                c_temp[1:-1] = c[2:] * alfa_i[1:-1] + betta_i[1:-1]
                c_temp[-1] = c_bound
                c_temp[0] = c_temp[1]
                return c_temp

        elif self.key == 'sphere':
            if self.key_sch == 'explicit':
                if dt> dr/(2 *self.diff_coef*porosity/tau_izv):  # diff_coef*(dt/dr + 2 * stab_cond) > 1:
                    sverka_method = 5
                    pass
                else:
                    for i in range(len(r)):
                        c_temp[1:-1] = c[1:-1] + self.diff_coef * dt * (
                                    (c[2:] - 2 * c[1:-1] + c[0:-2]) / dr ** 2 +  (c[2:] - c[0:-2]) / (
                                        r[i] * dr))  # явная разностная схема с ЦКР работает
                    c_temp[-1] = c_bound
                    c_temp[0] = c_temp[1]
                    return c_temp

            elif self.key_sch == 'implicit':
                for j in range(1, len(r)):
                    a = -self.diff_coef * dt / (dr) ** 2 - (1 / r[j]) * self.diff_coef * dt / dr
                    b = 1 + 2 * dt * self.diff_coef / (dr) ** 2
                    c_koef = -dt * self.diff_coef / (dr) ** 2 + (1 / r[j]) * self.diff_coef * dt / dr

                for i in range(1, len(l)):
                    alfa_i[i] = (-a) / (b + c_koef * alfa_i[i - 1])
                    betta_i[i] = (c[i] - c_koef * betta_i[i - 1]) / (b + c_koef * alfa_i[i - 1])

                alfa_i[-1] = 0
                betta_i[-1] = 0
                c_temp[1:-1] = c[2:] * alfa_i[1:-1] + betta_i[1:-1]
                c_temp[-1] = c_bound
                c_temp[0] = c_temp[1]
                return c_temp

    def fick_changed_fin(self, T, P, y_fick, c_changed, D_coef_ips_co2, D_coef_co2_ips, c_bound, y_bound, dr, dt, r):
        global sverka_method
        sverka_method = 0
        print('Граничное условие', y_bound)
        h_arr = []
        h_arr.append(0)
        for i in range(1, num_steps + 2):
            h_arr.append(R / num_steps + h_arr[i - 1])

        v = 1 #костыль для цилиндра

        c_coef_new = [None] * (num_steps + 2)
        a_coef_new = [None] * (num_steps + 2)
        b_coef_new = [None] * (num_steps + 2)
        alfa_new = [1] * (num_steps + 2)
        beta_new = [0] * (num_steps + 2)

        c_changed_fin = []
        c_changed_fin = np.copy(c_changed)
        y_fick_fin = []
        y_fick_fin = np.copy(y_fick)
        print('y_fick_fin', y_fick_fin)
        print('Список концентрации в новой функции', c_changed_fin)
        density_mixture = []
        M_ips_in_gel = []  # Мольная доля
        D_coef_mol_in_gel = []
        V_ips = []

        for i in range(0, num_steps + 2):
            M_ips_gel = M_co2 / 1000 * y_fick_fin[i] / (
                        (1 - y_fick_fin[i]) * M_ips / 1000 + y_fick_fin[i] * M_co2 / 1000)

            M_ips_in_gel.append(M_ips_gel)
            D_coef_mol_gels = D_coef_co2_ips ** M_ips_in_gel[i] * D_coef_ips_co2 ** (1 - M_ips_in_gel[i])

            D_coef_mol_in_gel.append(D_coef_mol_gels)


        for i in range(1, num_steps + 1):
            c_coef_new[i] = (D_coef_mol_in_gel[i] + D_coef_mol_in_gel[i - 1]) * target_material_porosity / tau_izv / 2 * (
                                    (h_arr[i] ** v + h_arr[i - 1] ** v) / 2) / (dr ** 2 * h_arr[i] ** v)

            a_coef_new[i] = (D_coef_mol_in_gel[i] + D_coef_mol_in_gel[
                i + 1]) * target_material_porosity / tau_izv / 2 * (
                                    (h_arr[i] ** v + h_arr[i + 1] ** v) / 2) / (dr ** 2 * h_arr[i] ** v)

            b_coef_new[i] = 1 / dt + a_coef_new[i] + c_coef_new[i]

            alfa_new[i] = a_coef_new[i] / (b_coef_new[i] - alfa_new[i - 1] * c_coef_new[i])
            beta_new[i] = (beta_new[i - 1] * c_coef_new[i] + y_fick[i] / dt) / (
                    b_coef_new[i] - alfa_new[i - 1] * c_coef_new[i])

        for i in range(num_steps, 0, -1):
            y_fick_fin[i] = alfa_new[i] * y_fick_fin[i + 1] + beta_new[i]

        y_fick_fin[-1] = y_bound
        y_fick_fin[0] = y_fick_fin[1]
        print('y_fick', y_fick_fin)

        density_mixture_num = None
        for i in range(0, num_steps + 2):
            V_ips_num = density_co2 * y_fick_fin[i] / (density_ips * (1 - y_fick_fin[i]) + density_co2 * y_fick_fin[i])

            density_mixture_num = density_ips * V_ips_num + density_co2 * (1 - V_ips_num)

            V_ips.append(V_ips_num)
            density_mixture.append(density_mixture_num)

        #density_mixture[-1] = density_co2
        density_mixture[0] = density_mixture[1]

        #V_ips[-1] = 0
        V_ips[0] = V_ips[1]

        for i in range(0, num_steps + 2):
            c_changed_fin[i] = y_fick_fin[i] * density_mixture[i]

        return y_fick_fin, c_changed_fin, density_mixture, V_ips
    def fick_mass(self, c, length, width, number_samples):
        m = 0.
        for i in range(1, len(c)):
            if self.key == 'sphere':
                m += c[i] * 4 / 3 * np.pi * ((i * dr) ** 3 - ((i - 1) * dr) ** 3)
            elif self.key == 'cyl':
                m += c[i] * length * np.pi * ((i * dr) ** 2 - ((i - 1) * dr) ** 2)
            elif self.key == 'one_dim':
                m += c[i] * ((2 * i * dr) - ((i - 1) * 2 * dr)) * length * width
        return m * number_samples


    def time_iteration(self, T, P, V_init_list,  c_init_list, c_changed_init, y_fick_init, D_coef_ips_co2, D_coef_co2_ips, volume, flowrate, n_t, dt, dr, number_samples):
        global method_value
        method_value = 0  # костыль для определения и не вылетания объёма аппарата
        residence_time = volume / flowrate

        c_app = np.zeros(n_t)
        mass_list = np.zeros(n_t)

        V_ips=np.zeros((n_t, len(V_init_list)))
        c_matrix = np.zeros((n_t, len(c_init_list)))
        c_matrix_changed = np.zeros((n_t, len(c_changed_init)))
        y_fick_changed = np.zeros((n_t, len(y_fick_init)))
        density_mix = np.zeros((n_t, len(y_fick_init)))

        c_matrix[0] = c_init_list
        c_matrix_changed[0] = c_changed_init
        y_fick_changed[0] = y_fick_init
        density_mix[0] = density_ips
        V_ips[0] = V_init_list#TODO доделать returnom из main. Сделать пересчёт на массу, а не концентрауию
        if self.key_sch == ('implicit' or 'explicit'):
            mass_list[0] = self.fick_mass(c_matrix[0], self.length, self.width, self.number_samples)
        elif self.key_sch == 'implicit modified':
            mass_list[0] = self.fick_mass(c_matrix_changed[0], self.length, self.width, self.number_samples)
        c_app[0] = 0.

        layout = [[psg.ProgressBar(n_t, orientation='h', expand_x=True, size=(20, 20), key='-PROG-')], [psg.Text('', key='-OUT-', enable_events=True, font=('Arial Bold', 16), justification='center', expand_x=True)]]
        window = psg.Window('Progress Bar', layout, size=(715, 150))

        for i in range(1, n_t):
            if i == 1:
                y_boundary = y_start

            y_bound = y_boundary
            c_bound = c_app[i - 1]

            event, values = window.read(timeout = 1)

            if self.key_sch == ('implicit' or 'explicit'):
                c_matrix[i] = self.fick_conc(T, P, c_matrix[i - 1], c_bound, dr, dt, r)

            elif self.key_sch == 'implicit modified':
                y_fick_changed[i], c_matrix_changed[i], density_mix[i], V_ips[i] = self.fick_changed_fin(T, P, y_fick_changed[i-1], c_matrix_changed[i - 1],D_coef_ips_co2, D_coef_co2_ips, c_bound, y_bound, dr, dt, r)
            print('Список мольных долей', V_ips[i])
            print('Концентрация', c_matrix_changed[i])
            if volume * load_perc < self.V_gels:
                method_value = 5
                pass

            else:
                if method_value != 5:
                    if self.key_sch == ('implicit' or 'explicit'):
                        mass_list[i] = self.fick_mass(c_matrix[i], self.length, self.width, self.number_samples)

                    elif self.key_sch == 'implicit modified':
                        mass_list[i] = self.fick_mass(c_matrix_changed[i], self.length, self.width, self.number_samples)

                    delta_mass = -1* (mass_list[i] - mass_list[i - 1])
                    c_app[i] = self.ideal_mixing(c_app[i - 1], 0, residence_time, dt, volume, delta_mass)

                    y_boundary = c_app[i] / density_mix[i][-1]
                    y_fick_changed[i][-1] = y_boundary
                    print('Концентрация в аппарате',c_app[i] ,'Граница', y_boundary,  'плотность', density_mix[i][-1])
            window['-PROG-'].update(current_count=i + 1)
            window['-OUT-'].update(str(i + 1) + f'из {n_t} итераций')


        window.close()
        if self.key_sch == ('implicit' or 'explicit'):
            return c_matrix, mass_list, c_app

        elif self.key_sch == 'implicit modified':
            return c_matrix_changed, mass_list, c_app

    def ideal_mixing(self, c, c_inlet, residence_time, dt, volume, delta_mass):
        c_mixing = c + dt / residence_time * (c_inlet - c) + dt * delta_mass / volume
        print('c_mix', c_mixing, 'delta_mass', delta_mass)
        return c_mixing

def main(T, P, width, length, height, volume, flowrate, dt, diff_coef, number_samples, value, key_sch, working, working_scheme):
    global n_t, R, dr, c_r, r, R, y_start
    c_init = density_ips * porosity
    R = height / 2  # meters
    dr = R / num_steps  # шаг по радиусу meters
    n_t = int(proc_time / dt) + 1  # количество шагов с учетом нулевого шага
    c_init_list = np.zeros(num_steps + 2)

    c_r = np.zeros(num_steps + 2)
    r = np.linspace(0, R, num_steps + 2)

    for i in range(num_steps + 2):
        c_init_list[i] = c_init
        if i == num_steps + 1:
            c_init_list[i] = 0


    object1 = scd_apparatus(T, P, volume, flowrate, width, length, height, diff_coef, value, key_sch, number_samples)
    object1.__str__()


    y_start = float(optimize.golden(object1.golden_method, maxiter=10000))  # массовая доля, кг/кгсм
    density_co2, D_coef_ips_co2, D_coef_co2_ips, dynamic_viscosity_ips, dynamic_viscosity_co2 = object1.density_co2_and_viscos(T, P)

    y_fick = [y_start] * (num_steps +2)     #создаю список массовых долей и наполняю его начальным условиями
    y_fick[-1] = y_start
    y_fick_init = y_fick

    V_ips_num = density_co2 * y_start / (density_ips * (1 - y_start) + density_co2 * y_start)
    V_ips = (num_steps+2) * [V_ips_num]
    V_ips[-1] = V_ips_num
    density_mixture = np.zeros(num_steps+2)

    for i in range(num_steps+2):
        density_mixture[i] = density_ips * V_ips[i] + density_co2 * (1 - V_ips[i])
        if i == num_steps + 1:
            density_mixture[i] = density_ips * V_ips[i] + density_co2 * (1 - V_ips[i])


    c_changed_init = np.zeros(num_steps + 2)
    for i in range(num_steps + 2):
        c_changed_init[i] = density_ips * y_start
        if i == num_steps + 1:
            #c_changed_init[i] = 0
            c_changed_init[i] = density_ips * y_start

    print('n_t:', n_t, 'proc_time:', proc_time, 'variable of item',value)
    time = np.linspace(0, proc_time, n_t)

    value = ['one_dim', 'cyl', 'sphere']
    key_sch = ['explicit', 'implicit', 'implicit modified']

    matrix_of_c, list_of_mass, c_app = object1.time_iteration(T, P, V_ips, c_init_list, c_changed_init, y_fick_init,
                                                              D_coef_ips_co2, D_coef_co2_ips, volume, flowrate, n_t, dt, dr, number_samples)

    return matrix_of_c, list_of_mass, c_app, time, i, r, method_value, sverka_method
