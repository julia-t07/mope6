import random, math, numpy
import scipy.stats

x1_min = 10
x1_max = 40
x2_min = 10
x2_max = 60
x3_min = 10
x3_max = 15
l = 1.73


x01 = (x1_max + x1_min) / 2
xl1 = l*(x1_max-x01)+x01
x02 = (x2_max + x2_min) / 2
xl2 = l*(x2_max-x02)+x02
x03 = (x3_max + x3_min) / 2
xl3 = l*(x3_max-x03)+x03



Xf = [[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],  # 1
      [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # 2
      [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # 3
      [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0],  # 4
      [1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 5
      [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],  # 6
      [1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],  # 7
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 8
      [-1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],  # 9
      [1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],  # 10
      [0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],  # 11
      [0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],  # 12
      [0, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929],  # 13
      [0, 0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929],  # 14
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 15
Gt = {1: 0.9065, 2: 0.7679, 3: 0.6841, 4: 0.6287, 5: 0.5892, 6: 0.5598, 7: 0.5365, 8: 0.5175, 9: 0.5017,
      10: 0.4884}
Gt2 = [(range(11, 17), 0.4366), (range(17, 37), 0.3720), (range(37, 145), 0.3093)]

def form_xnat():
    xnat = [[-1.0, -1.0, -1.0],  # 1
            [-1.0, -1.0, 1.0],  # 2
            [-1.0, 1.0, -1.0],  # 3
            [-1.0, 1.0, 1.0],  # 4
            [1.0, -1.0, -1.0],  # 5
            [1.0, -1.0, 1.0],  # 6
            [1.0, 1.0, -1.0],  # 7
            [1.0, 1.0, 1.0],  # 8
            [-1.73, 0, 0],  # 9
            [1.73, 0, 0],  # 10
            [0, -1.73, 0],  # 11
            [0, 1.73, 0],  # 12
            [0, 0, -1.73],  # 13
            [0, 0, 1.73],  # 14
            [0, 0, 0]]  # 15
    for i in range(len(xnat)):
        if xnat[i][0] == -1:
            xnat[i][0] = x1_min
        elif xnat[i][0] == 1:
            xnat[i][0] = x1_max
        elif xnat[i][0] == -1.73:
            xnat[i][0] = -xl1
        elif xnat[i][0] == 1.73:
            xnat[i][0] = xl1
        else:
            xnat[i][0] = x01
        # -------------------
        if xnat[i][1] == -1:
            xnat[i][1] = x2_min
        elif xnat[i][1] == 1:
            xnat[i][1] = x2_max
        elif xnat[i][1] == -1.73:
            xnat[i][1] = -xl2
        elif xnat[i][1] == 1.73:
            xnat[i][1] = xl2
        else:
            xnat[i][1] = x02
        # ---------------------
        if xnat[i][2] == -1:
            xnat[i][2] = x3_min
        elif xnat[i][2] == 1:
            xnat[i][2] = x3_max
        elif xnat[i][2] == -1.73:
            xnat[i][2] = -xl3
        elif xnat[i][2] == 1.73:
            xnat[i][2] = xl3
        else:
            xnat[i][2] = x03
    return xnat

def get_b(n, lmaty):

    xnat = form_xnat()

    print("X: ")
    for i in range(len(xnat)):
        print(xnat[i])
    print("-------------------------")
    xnm = [[xnat[i][j] for i in range(15)] for j in range(3)]
    
    a00 = [[],
           [xnm[0]], [xnm[1]], [xnm[2]],
           [xnm[0], xnm[1]],
           [xnm[0], xnm[2]],
           [xnm[1], xnm[2]],
           [xnm[0], xnm[1], xnm[2]],
           [xnm[0], xnm[0]],
           [xnm[1], xnm[1]],
           [xnm[2], xnm[2]]]
    
    def calcxi(n, listx):
        sumxi = 0
        for i in range(n):
            lsumxi = 1
            for j in range(len(listx)):
                lsumxi *= listx[j][i]
            sumxi += lsumxi
        return sumxi
    
    a0 = [15]
    for i in range(10):
        a0.append(calcxi(n, a00[i + 1]))
        
    a1 = [calcxi(n, a00[i] + a00[1]) for i in range(len(a00))]
    a2 = [calcxi(n, a00[i] + a00[2]) for i in range(len(a00))]
    a3 = [calcxi(n, a00[i] + a00[3]) for i in range(len(a00))]
    a4 = [calcxi(n, a00[i] + a00[4]) for i in range(len(a00))]
    a5 = [calcxi(n, a00[i] + a00[5]) for i in range(len(a00))]
    a6 = [calcxi(n, a00[i] + a00[6]) for i in range(len(a00))]
    a7 = [calcxi(n, a00[i] + a00[7]) for i in range(len(a00))]
    a8 = [calcxi(n, a00[i] + a00[8]) for i in range(len(a00))]
    a9 = [calcxi(n, a00[i] + a00[9]) for i in range(len(a00))]
    a10 = [calcxi(n, a00[i] + a00[10]) for i in range(len(a00))]

    a = numpy.array([[a0[0], a0[1], a0[2], a0[3], a0[4], a0[5],
                      a0[6], a0[7], a0[8], a0[9], a0[10]],
                     [a1[0], a1[1], a1[2], a1[3], a1[4], a1[5],
                      a1[6], a1[7], a1[8], a1[9], a1[10]],
                     [a2[0], a2[1], a2[2], a2[3], a2[4], a2[5],
                      a2[6], a2[7], a2[8], a2[9], a2[10]],
                     [a3[0], a3[1], a3[2], a3[3], a3[4], a3[5],
                      a3[6], a3[7], a3[8], a3[9], a3[10]],
                     [a4[0], a4[1], a4[2], a4[3], a4[4], a4[5],
                      a4[6], a4[7], a4[8], a4[9], a4[10]],
                     [a5[0], a5[1], a5[2], a5[3], a5[4], a5[5],
                      a5[6], a5[7], a5[8], a5[9], a5[10]],
                     [a6[0], a6[1], a6[2], a6[3], a6[4], a6[5],
                      a6[6], a6[7], a6[8], a6[9], a6[10]],
                     [a7[0], a7[1], a7[2], a7[3], a7[4], a7[5],
                      a7[6], a7[7], a7[8], a7[9], a7[10]],
                     [a8[0], a8[1], a8[2], a8[3], a8[4], a8[5],
                      a8[6], a8[7], a8[8], a8[9], a8[10]],
                     [a9[0], a9[1], a9[2], a9[3], a9[4], a9[5],
                      a9[6], a9[7], a9[8], a9[9], a9[10]],
                     [a10[0], a10[1], a10[2], a10[3], a10[4], a10[5],
                      a10[6], a10[7], a10[8], a10[9], a10[10]]])
    c0 = [calcxi(n, [lmaty])]
    for i in range(len(a00) - 1):
        c0.append(calcxi(n, a00[i + 1] + [lmaty]))
    c = numpy.array([c0[0], c0[1], c0[2], c0[3], c0[4], c0[5],
                     c0[6], c0[7], c0[8], c0[9], c0[10]])
    b = numpy.linalg.solve(a, c)

    return b

def gen_y(n, m):
    xnat = form_xnat()
    def f(x1, x2, x3):
        f = 3.6 + 8.8*x1 + 4.8*x2 + 5.1*x3
        f += 5.0 * x1 * x1 + 0.9 * x2 * x2 + 7.6 * x3 * x3
        f += 2.2 * x1 * x2 + 0.4 * x1 * x3 + 3.2 * x2 * x3 + 7.1 * x1 * x2 * x3
        return f

    Y = [[round(f(*xnat[i]) + random.randint(0, 10) - 5, 2) for j in range(m)] for i in range(n)]
    
    return Y

def cmb(arr):
    return [1, *arr,
           round(arr[0]*arr[1], 2),
           round(arr[0]*arr[2], 2),
           round(arr[1]*arr[2], 2),
           round(arr[0]*arr[1]*arr[2], 2),
           round(arr[0]*arr[0], 2),
           round(arr[1]*arr[1], 2),
           round(arr[2]*arr[2], 2)]

def func(num):
    N = 15
    m = num
    Y = gen_y(N, m)
    print("Y: ")
    for i in range(len(Y)):
        print(Y[i])
    print("-------------------------")
    # матриця Х
    Ys = [sum(Y[i])/m for i in range(N)]
    b_arr = get_b(N, Ys)
    b0 = b_arr[0]
    b1 = b_arr[1]
    b2 = b_arr[2]
    b3 = b_arr[3]
    b12 = b_arr[4]
    b13 = b_arr[5]
    b23 = b_arr[6]
    b123 = b_arr[7]
    b11 = b_arr[8]
    b22 = b_arr[9]
    b33 = b_arr[10]

    # нормоване рівняння
    print("Y={} + {}*x1 + {}*x2 \n"
          "+ {}*x3 +{}*x1x2 + {}*x1x3 \n"
          "+ {}*x2x3 + {}*x1x2x3 \n"
          "+ {}*x1^2 + {}*x2^2 \n"
          "+ {}*x3^2".format(b0, b1, b2, b3,b12, b13, b23, b123, b11, b22, b33))
    print("-------------------------")
    # перевірка
    print("Перевірка: ")
    xnat = form_xnat()

    for i in range(len(Xf)):
        print(round(sum([cmb(xnat[i])[j] * b_arr[j] for j in range(11)]), 2), "==", Ys[i])
    print("Результат збігається з середніми значеннями")
    print("-------------------------")
    

    # перевірка однорідності за Кохреном
    # дисперсії по рядках
    print("Критерій Кохрена")
    D = []
    Summa = 0
    for i in range(N):
        for j in range(m):
            Summa += pow((Y[i][j] - Ys[i]), 2)
        D.append(1 / m * Summa)
        Summa = 0

    Gp = max(D) / sum(D)
    print("Gp= ", Gp)
    f1 = m - 1
    f2 = N
    q = 0.05
    if m >= 11:
        for i in range(len(Gt2)):
            if m in Gt2[i][0]:
                crit = Gt2[i][1]
                break
    else:
        crit = Gt[f1]
    if Gp <= crit:
        print("Дисперсія однорідна")
        print(Gp, "<=", crit)
    else:
        print("Дисперсія не однорідна")
        m += 1
        print("M:", m)
        return func(m)
    print("-------------------------")
    # критерій Стьюдента
    print("Критерій Стьюдента")
    S2_b = sum(D) / N
    S2_betta = S2_b / (N * m)
    S_betta = math.sqrt(S2_betta)

    Xs = [[1.0,-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0],  # 1
          [1.0,-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # 2
          [1.0,-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],  # 3
          [1.0,-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0],  # 4
          [1.0,1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 5
          [1.0,1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],  # 6
          [1.0,1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],  # 7
          [1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 8
          [1.0,-1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],  # 9
          [1.0,1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],  # 10
          [1.0,0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],  # 11
          [1.0,0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],  # 12
          [1.0,0, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929],  # 13
          [1.0,0, 0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929],  # 14
          [1.0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # 15
    betta = []
    for i in range(N):
        s = 0
        for j in range(11):
            s += Ys[i] * Xs[i][j]
        betta.append(s / N)

    t = []
    for i in range(len(betta)):
        t.append(abs(betta[i]) / S_betta)

    f3 = f1 * f2
    print("f3=", f3)

    t_tabl = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)
    if t[i] < t_tabl:
        b_arr[i] = 0
        print(t[i], "<", t_tabl)
    print("-------------------------")

    for i in range(len(Ys)):
        print(round(sum([cmb(xnat[i])[j] * b_arr[j] for j in range(11)]), 2), "==", Ys[i])
    print("Нуль гіпотеза виконується")
    print("-------------------------")
    print("Критерій Фішера")
    # критерій Фішера
    d = 0
    for i in range(len(b_arr)):
        if b_arr[i] != 0:
            d += 1
    print("d=", d)
    f4 = N - d
    disp = []
    for i in Y:
        s = 0
        for k in range(m):
            s += (Ys[k] - i[k]) ** 2
        disp.append(s / m)
    S_ad = sum([(sum([cmb(xnat[i])[j] * b_arr[j] for j in range(11)]) - Ys[i]) ** 2 for i in range(N)])        
    S_ad = S_ad * m / (N - d)
    Fp = S_ad / sum(disp) / N
    print("S_ad= ", S_ad)
    print("Fp= {0} \n"
          "f3= {1} \n"
          "f4= {2}".format(Fp, f3, f4))
    print("-------------------------")
    Ft = scipy.stats.f.ppf(1 - q, f4, f3)
    while True:
        if (Fp < Ft):
            print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05")
            print("Значення критерію=", Ft)
            break
        else:
            print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
            print("Значення критерію=", Ft)
            return (func(m))


func(2)
