from matplotlib.pyplot import figure, show
from numpy import arange, sin,cos, pi




def square_func(x):
    if (x>-1 and x<1):
     return 1   
    elif (x<=1 or x>=1):
     return 0


N = input('Enter N ? ')
N = abs(int(N))
T_0=4
T_1=1
Summation = 400*[(2*T_1/T_0)] if N == 0 else 0
fig = figure(1)
t = arange(-2,2, 0.01)
ax1 = fig.add_subplot(111)

if N >0:
    Summation = 0




    for k in range(-N,0):
        Summation =  Summation + ((sin(pi*k/2))/(k*pi))*cos(k*2*pi*t/T_0)
         
    Summation = Summation + (2*T_1/T_0)

    for k in range(1,N+1):
        Summation =  Summation + ((sin(pi*k/2))/(k*pi))*cos(k*2*pi*t/T_0)

 
ax1.plot(t,Summation)
square = [square_func(i) for i in t]
ax1.plot(t,square,'r')
# ax1.plot(t, cos(2 * pi * t))
ax1.grid(True)
ax1.set_ylim((-.1, 1.2))
ax1.set_ylabel('Amplitude')
ax1.set_title('Fourier Series with N = '+str(N))
show()
